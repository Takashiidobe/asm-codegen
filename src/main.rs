use std::borrow::Cow::{self, Borrowed, Owned};
use std::env::args;
use std::error::Error;
use std::fs::read_to_string;

use asm_codegen::asm_interpreter::Codegen;
use asm_codegen::parser::Parse;
use asm_codegen::{lexer::Lexer, parser::Parser};

use rustyline::completion::FilenameCompleter;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::hint::HistoryHinter;
use rustyline::validate::MatchingBracketValidator;
use rustyline::{Completer, Helper, Hinter, Validator};
use rustyline::{CompletionType, Config, EditMode, Editor};

#[derive(Debug, Default, Clone, PartialEq, Copy)]
enum Mode {
    Asm,
    AsmFile,
    #[default]
    Repl,
}

#[derive(Helper, Completer, Hinter, Validator)]
struct MyHelper {
    #[rustyline(Completer)]
    completer: FilenameCompleter,
    highlighter: MatchingBracketHighlighter,
    #[rustyline(Validator)]
    validator: MatchingBracketValidator,
    #[rustyline(Hinter)]
    hinter: HistoryHinter,
    colored_prompt: String,
}

impl Highlighter for MyHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> Cow<'b, str> {
        if default {
            Borrowed(&self.colored_prompt)
        } else {
            Borrowed(prompt)
        }
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Owned("\x1b[1m".to_owned() + hint + "\x1b[m")
    }

    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, forced: bool) -> bool {
        self.highlighter.highlight_char(line, pos, forced)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = args().collect();
    let mut mode = Mode::default();
    if args.len() > 1 {
        match args[1].as_str() {
            "asm" => {
                mode = Mode::Asm;
            }
            _ => {
                mode = Mode::AsmFile;
            }
        }
    }

    if mode == Mode::AsmFile {
        let s = read_to_string(args[1].as_str()).expect("Could not find file");
        let mut lexer = Lexer::new(&s);
        let tokens = lexer.scan_tokens();
        let mut parser = Parser::new(tokens.clone());
        let stmts = parser.parse();
        let mut codegen = Codegen::new();
        let instructions = codegen.program(&stmts?);
        for instruction in instructions {
            println!("{}", instruction);
        }
        Ok(())
    } else {
        repl(mode)
    }
}

fn repl(mode: Mode) -> Result<(), Box<dyn Error>> {
    let config = Config::builder()
        .history_ignore_space(true)
        .completion_type(CompletionType::List)
        .edit_mode(EditMode::Vi)
        .build();
    let h = MyHelper {
        completer: FilenameCompleter::new(),
        highlighter: MatchingBracketHighlighter::new(),
        hinter: HistoryHinter::new(),
        colored_prompt: "".to_owned(),
        validator: MatchingBracketValidator::new(),
    };
    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(h));
    if rl.load_history("history.txt").is_err() {
        println!("No previous history.");
    }

    let mut count = 1;
    loop {
        let p = format!("{count}> ");
        rl.helper_mut().expect("No helper").colored_prompt = format!("\x1b[1;32m{p}\x1b[0m");
        let readline = rl.readline(&p);
        match readline {
            Ok(s) => {
                let mut lexer = Lexer::new(&s);
                let tokens = lexer.scan_tokens();
                let mut parser = Parser::new(tokens.clone());
                let stmts = parser.parse();
                match mode {
                    Mode::Asm => {
                        let mut codegen = Codegen::new();
                        let instructions = codegen.program(&stmts?);
                        for instruction in instructions {
                            println!("{}", instruction);
                        }
                    }
                    _ => unreachable!(),
                }
                rl.add_history_entry(s.as_str())?;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
        count += 1;
    }
    rl.append_history("history.txt")?;
    Ok(())
}
