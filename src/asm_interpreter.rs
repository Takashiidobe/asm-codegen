use std::borrow::Borrow as _;
use std::{collections::HashMap, fmt};

use crate::error::error;
use crate::token::{ObjType, TokenType};
use crate::{
    expr::Expr,
    stmt::Stmt,
    token::{Object, Token},
};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Codegen {
    depth: i64,
    instructions: Vec<AsmInstruction>,
    float_count: u64,
    str_count: u64,
    stack_offset: i64,
    pub vars: HashMap<String, (i64, ObjType)>,
    var_offset: i64,
    pub functions: HashMap<String, ObjType>,
    pub labels: HashMap<String, String>,
    label_count: u64,
    strings: Vec<AsmInstruction>,
    floats: Vec<AsmInstruction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Reg {
    Rax,
    Rsp,
    Rbp,
    Rbx,
    Rsi,
    Rdi,
    Rdx,
    Rcx,
    R8,
    R9,
    Al,
    Rip,
    Xmm0,
    Xmm1,
    Ecx,
    Eax,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Address {
    Reg(Reg),
    Label(String),
    Immediate(i64),
    Indirect(Reg),
    IndirectDouble(Reg, Reg),
    IndirectOffset(i64, Reg),
    LabelOffset(String, Reg),
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Address::Reg(reg) => f.write_fmt(format_args!("{}", reg)),
            Address::Label(label) => f.write_fmt(format_args!("$.{}", label)),
            Address::Immediate(num) => f.write_fmt(format_args!("${}", num)),
            Address::Indirect(reg) => f.write_fmt(format_args!("({})", reg)),
            Address::IndirectOffset(offset, reg) => {
                f.write_fmt(format_args!("{}({})", offset, reg))
            }
            Address::LabelOffset(offset, reg) => f.write_fmt(format_args!("{}({})", offset, reg)),
            Address::IndirectDouble(r1, r2) => f.write_fmt(format_args!("({},{})", r1, r2)),
        }
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Reg::Rax => "%rax",
            Reg::Rsp => "%rsp",
            Reg::Rbp => "%rbp",
            Reg::Rsi => "%rsi",
            Reg::Rdi => "%rdi",
            Reg::Rdx => "%rdx",
            Reg::Al => "%al",
            Reg::Rcx => "%rcx",
            Reg::R8 => "%r8",
            Reg::R9 => "%r9",
            Reg::Rip => "%rip",
            Reg::Xmm0 => "%xmm0",
            Reg::Xmm1 => "%xmm1",
            Reg::Rbx => "%rbx",
            Reg::Ecx => "%ecx",
            Reg::Eax => "%eax",
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AsmInstruction {
    Section(String),
    Variable(String, Option<String>),
    Label(String),
    Xor(Address, Reg),
    Push(Reg),
    Pop(Reg),
    Or(Address, Reg),
    And(Address, Reg),
    Call(String),
    Lea(Address, Address),
    Mov(Address, Address),
    Movq(Address, Address),
    Movb(Address, Address),
    Movsd(Address, Address),
    Movsbl(Address, Address),
    Movzbl(Address, Address),
    Movzb(Address, Address),
    Setae(Reg),
    Sete(Reg),
    Setb(Reg),
    Setbe(Reg),
    Setne(Reg),
    Setl(Reg),
    Setle(Reg),
    Setg(Reg),
    Setge(Reg),
    Setp(Reg),
    Setnp(Reg),
    Seta(Reg),
    Cqo,
    Ret,
    Test(Address, Address),
    Cmp(Address, Reg),
    Ucomisd(Address, Reg),
    Add(Address, Reg),
    Addsd(Reg, Reg),
    Sub(Address, Reg),
    Subsd(Reg, Reg),
    IMul(Address, Reg),
    Mulsd(Reg, Reg),
    IDiv(Address, Reg),
    Divsd(Reg, Reg),
    Neg(Reg),
    Je(String),
    Jne(String),
    Jmp(String),
    Jz(String),
    Byte(u8),
    Leave,
    Comment(String),
    Asciz(String),
    Double(f64),
    Xorpd(Address, Address),
}

impl fmt::Display for AsmInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use AsmInstruction::*;
        match self {
            Section(section) => f.write_fmt(format_args!(".{}", section)),
            Variable(var, val) => match val {
                Some(v) => f.write_fmt(format_args!("  .{} {}", var, v)),
                None => f.write_fmt(format_args!("  .{}", var)),
            },
            Lea(left, right) => f.write_fmt(format_args!("  lea {}, {}", left, right)),
            Label(label) => f.write_fmt(format_args!("{}:", label)),
            Or(left, right) => f.write_fmt(format_args!("  or {}, {}", left, right)),
            And(left, right) => f.write_fmt(format_args!("  and {}, {}", left, right)),
            Xor(left, right) => f.write_fmt(format_args!("  xor {}, {}", left, right)),
            Push(reg) => f.write_fmt(format_args!("  push {}", reg)),
            Pop(reg) => f.write_fmt(format_args!("  pop {}", reg)),
            Sete(reg) => f.write_fmt(format_args!("  sete {}", reg)),
            Setne(reg) => f.write_fmt(format_args!("  setne {}", reg)),
            Setb(reg) => f.write_fmt(format_args!("  setb {}", reg)),
            Setbe(reg) => f.write_fmt(format_args!("  setbe {}", reg)),
            Setl(reg) => f.write_fmt(format_args!("  setl {}", reg)),
            Setle(reg) => f.write_fmt(format_args!("  setle {}", reg)),
            Setg(reg) => f.write_fmt(format_args!("  setg {}", reg)),
            Setge(reg) => f.write_fmt(format_args!("  setge {}", reg)),
            Setp(reg) => f.write_fmt(format_args!("  setp {}", reg)),
            Setnp(reg) => f.write_fmt(format_args!("  setnp {}", reg)),
            Seta(reg) => f.write_fmt(format_args!("  seta {}", reg)),
            Setae(reg) => f.write_fmt(format_args!("  setae {}", reg)),
            Ret => f.write_fmt(format_args!("  ret")),
            Call(fn_name) => f.write_fmt(format_args!("  call {}", fn_name)),
            Test(left, right) => f.write_fmt(format_args!("  test {}, {}", left, right)),
            Cmp(left, right) => f.write_fmt(format_args!("  cmp {}, {}", left, right)),
            Ucomisd(left, right) => f.write_fmt(format_args!("  ucomisd {}, {}", left, right)),
            Mov(left, right) => f.write_fmt(format_args!("  mov {}, {}", left, right)),
            Movq(left, right) => f.write_fmt(format_args!("  movq {}, {}", left, right)),
            Movb(left, right) => f.write_fmt(format_args!("  movb {}, {}", left, right)),
            Movsd(left, right) => f.write_fmt(format_args!("  movsd {}, {}", left, right)),
            Movsbl(left, right) => f.write_fmt(format_args!("  movsbl {}, {}", left, right)),
            Movzbl(left, right) => f.write_fmt(format_args!("  movzbl {}, {}", left, right)),
            Movzb(left, right) => f.write_fmt(format_args!("  movzb {}, {}", left, right)),
            Cqo => f.write_fmt(format_args!("  cqo")),
            Add(left, right) => f.write_fmt(format_args!("  add {}, {}", left, right)),
            Addsd(left, right) => f.write_fmt(format_args!("  addsd {}, {}", left, right)),
            Sub(left, right) => f.write_fmt(format_args!("  sub {}, {}", left, right)),
            Subsd(left, right) => f.write_fmt(format_args!("  subsd {}, {}", left, right)),
            IMul(left, right) => f.write_fmt(format_args!("  imul {}, {}", left, right)),
            Mulsd(left, right) => f.write_fmt(format_args!("  mulsd {}, {}", left, right)),
            IDiv(left, right) => f.write_fmt(format_args!("  idiv {}, {}", left, right)),
            Divsd(left, right) => f.write_fmt(format_args!("  divsd {}, {}", left, right)),
            Neg(reg) => f.write_fmt(format_args!("  neg {}", reg)),
            Jne(reg) => f.write_fmt(format_args!("  jne {}", reg)),
            Je(reg) => f.write_fmt(format_args!("  je {}", reg)),
            Jz(reg) => f.write_fmt(format_args!("  jz {}", reg)),
            Jmp(reg) => f.write_fmt(format_args!("  jmp {}", reg)),
            Byte(b) => f.write_fmt(format_args!("  .byte {}", b)),
            Leave => f.write_fmt(format_args!("  leave")),
            Comment(comment) => f.write_fmt(format_args!("  # {}", comment)),
            Asciz(bytes) => f.write_fmt(format_args!("  .asciz \"{}\"", bytes)),
            Double(float) => f.write_fmt(format_args!("  .double {}", float)),
            Xorpd(left, right) => f.write_fmt(format_args!("  xorpd {}, {}", left, right)),
        }
    }
}

impl Codegen {
    pub fn new() -> Self {
        Self {
            var_offset: -32 * 8,
            ..Default::default()
        }
    }

    pub fn program(&mut self, stmts: &[Stmt]) -> Vec<AsmInstruction> {
        let mut body = vec![];
        let mut functions = vec![];
        for stmt in stmts {
            match stmt {
                Stmt::Function { .. } => {
                    functions.extend(self.stmt(stmt));
                }
                _ => {
                    body.extend(self.stmt(stmt));
                }
            }
        }
        let size = (self.var_offset.abs() + 2) as usize;
        let mut program = vec![AsmInstruction::Variable("text".to_string(), None)];
        let prologue = self.prologue(size);
        let epilogue = self.epilogue();

        program.extend(functions);
        program.extend(prologue);
        program.extend(body);
        program.extend(epilogue);
        if !self.strings.is_empty() || !self.floats.is_empty() {
            program.push(AsmInstruction::Variable("data".to_string(), None));
        }
        if !self.strings.is_empty() {
            program.extend(self.strings.clone());
        }
        if !self.floats.is_empty() {
            program.extend(self.floats.clone());
        }
        program
    }

    // return a 16 byte aligned stack for rsp
    // align the number to a multiple
    // eg. 3 8 byte items aligned to a 16 byte boundary would return 32.
    fn align_to(&self, size: usize, multiple: usize) -> usize {
        size.div_ceil(multiple) * multiple
    }

    fn prologue(&mut self, size: usize) -> Vec<AsmInstruction> {
        vec![
            // Str concat
            AsmInstruction::Label("str_concat".to_string()),
            AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::IndirectOffset(-16, Reg::Rbp),
            ),
            AsmInstruction::Mov(
                Address::Reg(Reg::Rdi),
                Address::IndirectOffset(-8, Reg::Rbp),
            ),
            AsmInstruction::Call("strlen".to_string()),
            AsmInstruction::Mov(Address::Reg(Reg::Rax), Address::Reg(Reg::Rbx)),
            AsmInstruction::Mov(
                Address::IndirectOffset(-16, Reg::Rbp),
                Address::Reg(Reg::Rdi),
            ),
            AsmInstruction::Call("strlen".to_string()),
            AsmInstruction::Add(Address::Reg(Reg::Rbx), Reg::Rax),
            AsmInstruction::Add(Address::Immediate(1), Reg::Rax),
            AsmInstruction::Mov(Address::Reg(Reg::Rax), Address::Reg(Reg::Rdi)),
            AsmInstruction::Call("malloc".to_string()),
            AsmInstruction::Mov(
                Address::IndirectOffset(-8, Reg::Rbp),
                Address::Reg(Reg::Rsi),
            ),
            AsmInstruction::Mov(Address::Reg(Reg::Rax), Address::Reg(Reg::Rdi)),
            AsmInstruction::Call("strcpy".to_string()),
            AsmInstruction::Mov(
                Address::IndirectOffset(-16, Reg::Rbp),
                Address::Reg(Reg::Rsi),
            ),
            AsmInstruction::Mov(Address::Reg(Reg::Rax), Address::Reg(Reg::Rdi)),
            AsmInstruction::Call("strcat".to_string()),
            AsmInstruction::Ret,
            // StrEq
            AsmInstruction::Label("str_eq".to_string()),
            AsmInstruction::Mov(
                Address::Reg(Reg::Rdi),
                Address::IndirectOffset(-8, Reg::Rbp),
            ),
            AsmInstruction::Mov(
                Address::Reg(Reg::Rsi),
                Address::IndirectOffset(-16, Reg::Rbp),
            ),
            AsmInstruction::Label("str_eq.1".to_string()),
            AsmInstruction::Mov(
                Address::IndirectOffset(-8, Reg::Rbp),
                Address::Reg(Reg::Rax),
            ),
            AsmInstruction::Movsbl(Address::Indirect(Reg::Rax), Address::Reg(Reg::Ecx)),
            AsmInstruction::Xor(Address::Reg(Reg::Eax), Reg::Eax),
            AsmInstruction::Cmp(Address::Immediate(0), Reg::Ecx),
            AsmInstruction::Movb(
                Address::Reg(Reg::Al),
                Address::IndirectOffset(-17, Reg::Rbp),
            ),
            AsmInstruction::Je("str_eq.3".to_string()),
            AsmInstruction::Mov(
                Address::IndirectOffset(-8, Reg::Rbp),
                Address::Reg(Reg::Rax),
            ),
            AsmInstruction::Movsbl(Address::Indirect(Reg::Rax), Address::Reg(Reg::Eax)),
            AsmInstruction::Mov(
                Address::IndirectOffset(-16, Reg::Rbp),
                Address::Reg(Reg::Rcx),
            ),
            AsmInstruction::Movsbl(Address::Indirect(Reg::Rcx), Address::Reg(Reg::Ecx)),
            AsmInstruction::Cmp(Address::Reg(Reg::Ecx), Reg::Eax),
            AsmInstruction::Sete(Reg::Al),
            AsmInstruction::Movb(
                Address::Reg(Reg::Al),
                Address::IndirectOffset(-17, Reg::Rbp),
            ),
            AsmInstruction::Label("str_eq.3".to_string()),
            AsmInstruction::Movb(
                Address::IndirectOffset(-17, Reg::Rbp),
                Address::Reg(Reg::Al),
            ),
            AsmInstruction::Test(Address::Immediate(1), Address::Reg(Reg::Al)),
            AsmInstruction::Jne("str_eq.4".to_string()),
            AsmInstruction::Jmp("str_eq.5".to_string()),
            AsmInstruction::Label("str_eq.4".to_string()),
            AsmInstruction::Mov(
                Address::IndirectOffset(-8, Reg::Rbp),
                Address::Reg(Reg::Rax),
            ),
            AsmInstruction::Add(Address::Immediate(1), Reg::Rax),
            AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::IndirectOffset(-8, Reg::Rbp),
            ),
            AsmInstruction::Mov(
                Address::IndirectOffset(-16, Reg::Rbp),
                Address::Reg(Reg::Rax),
            ),
            AsmInstruction::Add(Address::Immediate(1), Reg::Rax),
            AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::IndirectOffset(-16, Reg::Rbp),
            ),
            AsmInstruction::Jmp("str_eq.1".to_string()),
            AsmInstruction::Label("str_eq.5".to_string()),
            AsmInstruction::Mov(
                Address::IndirectOffset(-8, Reg::Rbp),
                Address::Reg(Reg::Rax),
            ),
            AsmInstruction::Movsbl(Address::Indirect(Reg::Rax), Address::Reg(Reg::Eax)),
            AsmInstruction::Mov(
                Address::IndirectOffset(-16, Reg::Rbp),
                Address::Reg(Reg::Rcx),
            ),
            AsmInstruction::Movsbl(Address::Indirect(Reg::Rcx), Address::Reg(Reg::Ecx)),
            AsmInstruction::Cmp(Address::Reg(Reg::Ecx), Reg::Eax),
            AsmInstruction::Sete(Reg::Al),
            AsmInstruction::And(Address::Immediate(1), Reg::Al),
            AsmInstruction::Movzb(Address::Reg(Reg::Al), Address::Reg(Reg::Rax)),
            AsmInstruction::Ret,
            // Prologue
            AsmInstruction::Variable("globl".to_string(), Some("main".to_string())),
            AsmInstruction::Label("main".to_string()),
            AsmInstruction::Push(Reg::Rbp),
            AsmInstruction::Mov(Address::Reg(Reg::Rsp), Address::Reg(Reg::Rbp)),
            AsmInstruction::Sub(Address::Immediate(self.align_to(size, 16) as i64), Reg::Rsp),
        ]
    }

    fn epilogue(&mut self) -> Vec<AsmInstruction> {
        vec![
            AsmInstruction::Xor(Address::Reg(Reg::Rax), Reg::Rax),
            AsmInstruction::Leave,
            AsmInstruction::Ret,
            AsmInstruction::Variable("section".to_string(), Some(".rodata".to_string())),
            AsmInstruction::Label(".format_i64".to_string()),
            AsmInstruction::Variable("string".to_string(), Some("\"%d\\n\"".to_string())),
            AsmInstruction::Label(".format_f64".to_string()),
            AsmInstruction::Variable("string".to_string(), Some("\"%f\\n\"".to_string())),
            AsmInstruction::Label(".nil_string".to_string()),
            AsmInstruction::Variable("string".to_string(), Some("\"nil\"".to_string())),
            AsmInstruction::Label(".true_string".to_string()),
            AsmInstruction::Variable("string".to_string(), Some("\"true\"".to_string())),
            AsmInstruction::Label(".false_string".to_string()),
            AsmInstruction::Variable("string".to_string(), Some("\"false\"".to_string())),
            AsmInstruction::Label("neg_mask".to_string()),
            AsmInstruction::Variable("quad".to_string(), Some("0x8000000000000000".to_string())),
        ]
    }

    fn push(&mut self) -> AsmInstruction {
        self.depth += 1;
        AsmInstruction::Push(Reg::Rax)
    }

    fn pop(&mut self, reg: Reg) -> AsmInstruction {
        self.depth -= 1;
        AsmInstruction::Pop(reg)
    }

    // make it so every expr returns a vector of results
    fn stmt(&mut self, stmt: &Stmt) -> Vec<AsmInstruction> {
        match stmt {
            Stmt::Expr { expr } => self.expr(expr).0,
            Stmt::Var { name, initializer } => {
                let (mut res, r_type) = if let Some(init) = initializer {
                    self.expr(init)
                } else {
                    self.expr(&Expr::Literal { value: Object::Nil })
                };

                let offset = self.new_var_offset();

                self.vars.insert(name.lexeme.clone(), (offset, r_type));

                if r_type == ObjType::Float {
                    res.push(AsmInstruction::Movsd(
                        Address::Reg(Reg::Xmm0),
                        Address::IndirectOffset(offset, Reg::Rbp),
                    ));
                    res.push(AsmInstruction::Xorpd(
                        Address::Reg(Reg::Xmm0),
                        Address::Reg(Reg::Xmm0),
                    ));
                } else {
                    res.push(AsmInstruction::Mov(
                        Address::Reg(Reg::Rax),
                        Address::IndirectOffset(offset, Reg::Rbp),
                    ));
                    res.push(AsmInstruction::Xor(Address::Reg(Reg::Rax), Reg::Rax));
                }

                res
            }
            Stmt::Block { stmts } => stmts.iter().flat_map(|x| self.stmt(x)).collect(),
            Stmt::Print { expr } => {
                let (mut expr_instruct, obj_type) = self.expr(expr);
                match obj_type {
                    ObjType::String => {
                        expr_instruct.extend(vec![
                            AsmInstruction::Mov(Address::Reg(Reg::Rax), Address::Reg(Reg::Rdi)),
                            AsmInstruction::Call("puts".to_string()),
                        ]);
                        expr_instruct
                    }
                    ObjType::Integer => {
                        expr_instruct.extend(vec![
                            AsmInstruction::Mov(Address::Reg(Reg::Rax), Address::Reg(Reg::Rsi)),
                            AsmInstruction::Mov(
                                Address::Label("format_i64".to_string()),
                                Address::Reg(Reg::Rdi),
                            ),
                            AsmInstruction::Call("printf".to_string()),
                        ]);
                        expr_instruct
                    }
                    ObjType::Bool => {
                        let count = self.get_label();
                        expr_instruct.extend(vec![
                            AsmInstruction::Cmp(Address::Immediate(0), Reg::Rax),
                            AsmInstruction::Je(format!(".L.else.{}", count)),
                            AsmInstruction::Mov(
                                Address::Label("true_string".to_string()),
                                Address::Reg(Reg::Rdi),
                            ),
                            AsmInstruction::Jmp(format!(".L.end.{}", count)),
                            AsmInstruction::Label(format!(".L.else.{}", count)),
                            AsmInstruction::Mov(
                                Address::Label("false_string".to_string()),
                                Address::Reg(Reg::Rdi),
                            ),
                            AsmInstruction::Label(format!(".L.end.{}", count)),
                            AsmInstruction::Call("puts".to_string()),
                        ]);
                        expr_instruct
                    }
                    ObjType::Float => {
                        expr_instruct.extend(vec![
                            AsmInstruction::Mov(
                                Address::Label("format_f64".to_string()),
                                Address::Reg(Reg::Rdi),
                            ),
                            AsmInstruction::Movb(Address::Immediate(1), Address::Reg(Reg::Al)),
                            AsmInstruction::Call("printf".to_string()),
                        ]);
                        expr_instruct
                    }
                    ObjType::Nil => {
                        expr_instruct.extend(vec![
                            AsmInstruction::Mov(
                                Address::Label("nil_string".to_string()),
                                Address::Reg(Reg::Rdi),
                            ),
                            AsmInstruction::Call("puts".to_string()),
                        ]);
                        expr_instruct
                    }
                    ObjType::Array => {
                        // TODO: Print array
                        expr_instruct.extend(vec![]);
                        expr_instruct
                    }
                }
            }
            Stmt::Function {
                name,
                params,
                body,
                return_type,
            } => {
                self.functions.insert(name.lexeme.clone(), *return_type);
                let mut res = vec![AsmInstruction::Label(name.to_string())];

                for (i, (param, r_type)) in params.iter().enumerate() {
                    let offset = (i as i64 + 1) * -8;
                    self.vars.insert(param.lexeme.clone(), (offset, *r_type));
                }

                for stmt in body {
                    res.extend(self.stmt(stmt));
                }

                res.push(AsmInstruction::Ret);
                res
            }
            Stmt::Return { value, .. } => {
                let mut instructions = vec![];

                instructions.extend(if let Some(val) = value {
                    self.expr(val).0
                } else {
                    self.expr(&Expr::Literal { value: Object::Nil }).0
                });
                instructions.extend(vec![AsmInstruction::Ret]);
                instructions
            }
            Stmt::If { cond, then, r#else } => {
                let count = self.get_label();
                let mut res = self.expr(cond).0;
                res.push(AsmInstruction::Cmp(Address::Immediate(0), Reg::Rax));
                res.push(AsmInstruction::Je(format!(".L.else.{}", count)));
                res.extend(self.stmt(then));
                res.push(AsmInstruction::Jmp(format!(".L.end.{}", count)));
                res.push(AsmInstruction::Label(format!(".L.else.{}", count)));
                if let Some(else_branch) = r#else.borrow() {
                    res.extend(self.stmt(else_branch));
                }
                res.push(AsmInstruction::Label(format!(".L.end.{}", count)));
                res
            }
            Stmt::While { cond, body } => {
                let count = self.get_label();

                let mut res = vec![];

                res.push(AsmInstruction::Label(format!(".L.begin.{}", count)));
                res.extend(self.expr(cond).0);
                res.push(AsmInstruction::Cmp(Address::Immediate(0), Reg::Rax));
                res.push(AsmInstruction::Je(format!(".L.end.{}", count)));
                res.extend(self.stmt(body));
                res.push(AsmInstruction::Jmp(format!(".L.begin.{}", count)));
                res.push(AsmInstruction::Label(format!(".L.end.{}", count)));

                res
            }
        }
    }

    // every expr should return a Vec<AsmInstruction>
    fn expr(&mut self, expr: &Expr) -> (Vec<AsmInstruction>, ObjType) {
        match expr {
            Expr::Binary { left, op, right } => match op {
                Token {
                    r#type: TokenType::Plus,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(right, left);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Add(Address::Reg(Reg::Rdi), Reg::Rax));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Addsd(Reg::Xmm0, Reg::Xmm1));
                        }
                        ObjType::String => {
                            res.push(AsmInstruction::Call("str_concat".to_string()));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call '{op}' on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, r_type)
                }
                Token {
                    r#type: TokenType::Minus,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Sub(Address::Reg(Reg::Rdi), Reg::Rax));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Subsd(Reg::Xmm0, Reg::Xmm1));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call '{op}' on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, r_type)
                }
                Token {
                    r#type: TokenType::Star,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::IMul(Address::Reg(Reg::Rdi), Reg::Rax));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Mulsd(Reg::Xmm0, Reg::Xmm1));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call '{op}' on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, r_type)
                }
                Token {
                    r#type: TokenType::Slash,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Cqo);
                            res.push(AsmInstruction::IDiv(Address::Reg(Reg::Rdi), Reg::Rax));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Divsd(Reg::Xmm0, Reg::Xmm1));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call {op} on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, r_type)
                }
                Token {
                    r#type: TokenType::EqualEqual,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                            res.push(AsmInstruction::Sete(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Ucomisd(Address::Reg(Reg::Xmm0), Reg::Xmm1));
                            res.push(AsmInstruction::Sete(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        ObjType::String => {
                            res.push(AsmInstruction::Call("str_eq".to_string()));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call {op} on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::BangEqual,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                            res.push(AsmInstruction::Setne(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Ucomisd(Address::Reg(Reg::Xmm0), Reg::Xmm1));
                            res.push(AsmInstruction::Setne(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        ObjType::String => {
                            res.push(AsmInstruction::Call("str_eq".to_string()));
                            res.push(AsmInstruction::Cmp(Address::Immediate(0), Reg::Rax));
                            res.push(AsmInstruction::Setne(Reg::Al));
                            res.push(AsmInstruction::Xor(Address::Immediate(-1), Reg::Al));
                            res.push(AsmInstruction::And(Address::Immediate(1), Reg::Al));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call {op} on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::Less,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                            res.push(AsmInstruction::Setl(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Ucomisd(Address::Reg(Reg::Xmm0), Reg::Xmm1));
                            res.push(AsmInstruction::Setb(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call {op} on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::LessEqual,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                            res.push(AsmInstruction::Setle(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Ucomisd(Address::Reg(Reg::Xmm0), Reg::Xmm1));
                            res.push(AsmInstruction::Setbe(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call {op} on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::Greater,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                            res.push(AsmInstruction::Setg(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Ucomisd(Address::Reg(Reg::Xmm0), Reg::Xmm1));
                            res.push(AsmInstruction::Seta(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call {op} on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::GreaterEqual,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    match r_type {
                        ObjType::Integer => {
                            res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                            res.push(AsmInstruction::Setge(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Ucomisd(Address::Reg(Reg::Xmm0), Reg::Xmm1));
                            res.push(AsmInstruction::Setae(Reg::Al));
                            res.push(AsmInstruction::Movzb(
                                Address::Reg(Reg::Al),
                                Address::Reg(Reg::Rax),
                            ));
                        }
                        _ => error(
                            op.line,
                            &format!("Cannot call {op} on operands {left:?}, {right:?}"),
                        ),
                    }
                    (res, ObjType::Bool)
                }
                _ => todo!(),
            },
            Expr::Assign { name, expr } => {
                let (mut res, r_type) = self.expr(expr);

                let offset = self.vars.get(&name.lexeme);
                if let Some((off, _)) = offset {
                    if r_type == ObjType::Float {
                        res.push(AsmInstruction::Movsd(
                            Address::Reg(Reg::Xmm0),
                            Address::IndirectOffset(*off, Reg::Rbp),
                        ));
                        res.push(AsmInstruction::Xorpd(
                            Address::Reg(Reg::Xmm0),
                            Address::Reg(Reg::Xmm0),
                        ));
                    } else {
                        res.push(AsmInstruction::Mov(
                            Address::Reg(Reg::Rax),
                            Address::IndirectOffset(*off, Reg::Rbp),
                        ));
                        res.push(AsmInstruction::Xor(Address::Reg(Reg::Rax), Reg::Rax));
                    }

                    self.vars.insert(name.lexeme.clone(), (*off, r_type));
                } else {
                    error(name.line, "Var {name} was not previously defined");
                }

                (res, r_type)
            }
            Expr::Var { name } => {
                if let Some((offset, r_type)) = self.vars.get(&name.lexeme) {
                    (
                        if *r_type == ObjType::Float {
                            vec![AsmInstruction::Movsd(
                                Address::IndirectOffset(*offset, Reg::Rbp),
                                Address::Reg(Reg::Xmm0),
                            )]
                        } else {
                            vec![AsmInstruction::Mov(
                                Address::IndirectOffset(*offset, Reg::Rbp),
                                Address::Reg(Reg::Rax),
                            )]
                        },
                        *r_type,
                    )
                } else {
                    panic!("Var {name} was not defined");
                }
            }
            Expr::Logical { .. } => todo!(),
            Expr::Unary { op, expr } => {
                if let Token {
                    r#type: TokenType::Minus,
                    ..
                } = op
                {
                    let (mut res, r_type) = self.expr(expr);
                    if r_type == ObjType::Integer {
                        res.push(AsmInstruction::Neg(Reg::Rax));
                    } else if r_type == ObjType::Float {
                        res.push(AsmInstruction::Movq(
                            Address::LabelOffset("neg_mask".to_string(), Reg::Rip),
                            Address::Reg(Reg::Xmm1),
                        ));
                        res.push(AsmInstruction::Xorpd(
                            Address::Reg(Reg::Xmm1),
                            Address::Reg(Reg::Xmm0),
                        ));
                    }
                    return (res, r_type);
                }
                panic!("Op: {op} cannot be applied to expr: {expr:?}")
            }
            Expr::Literal { value } => match value {
                Object::String(s) => {
                    let label = self.new_str_label();
                    self.strings.push(AsmInstruction::Label(label.clone()));
                    self.strings.push(AsmInstruction::Asciz(s.to_string()));
                    (
                        vec![AsmInstruction::Lea(
                            Address::LabelOffset(label, Reg::Rip),
                            Address::Reg(Reg::Rax),
                        )],
                        ObjType::String,
                    )
                }
                Object::Integer(i) => (
                    vec![AsmInstruction::Mov(
                        Address::Immediate(*i),
                        Address::Reg(Reg::Rax),
                    )],
                    ObjType::Integer,
                ),
                Object::Float(f) => {
                    let label = self.new_float_label();
                    self.floats.push(AsmInstruction::Label(label.clone()));
                    self.floats.push(AsmInstruction::Double(*f));
                    (
                        vec![AsmInstruction::Movsd(
                            Address::LabelOffset(label, Reg::Rip),
                            Address::Reg(Reg::Xmm0),
                        )],
                        ObjType::Float,
                    )
                }
                Object::Identifier(_) => todo!(),
                Object::Array(arr) => {
                    let mut res = vec![];
                    for item in arr {
                        let offset = self.new_var_offset();
                        let (evaled_expr, r_type) = self.expr(item);
                        res.extend(evaled_expr);
                        match r_type {
                            ObjType::Bool | ObjType::Nil | ObjType::String | ObjType::Integer => {
                                res.push(AsmInstruction::Mov(
                                    Address::Reg(Reg::Rax),
                                    Address::IndirectOffset(offset, Reg::Rbp),
                                ));
                            }
                            ObjType::Array => todo!(),
                            ObjType::Float => todo!(),
                        }
                    }
                    (res, ObjType::Array)
                }
                Object::Bool(b) => (
                    vec![AsmInstruction::Mov(
                        Address::Immediate(if *b { 1 } else { 0 }),
                        Address::Reg(Reg::Rax),
                    )],
                    ObjType::Bool,
                ),
                Object::Nil => (
                    vec![AsmInstruction::Mov(
                        Address::Immediate(0),
                        Address::Reg(Reg::Rax),
                    )],
                    ObjType::Nil,
                ),
            },
            Expr::Grouping { expr } => self.expr(expr),
            Expr::Call {
                callee, arguments, ..
            } => {
                let mut res = vec![];
                for (i, arg) in arguments.iter().enumerate() {
                    let (evaled_arg, r_type) = self.expr(arg);
                    res.extend(evaled_arg);
                    match r_type {
                        ObjType::String | ObjType::Bool | ObjType::Nil | ObjType::Integer => {
                            res.push(AsmInstruction::Mov(
                                Address::Reg(Reg::Rax),
                                Address::IndirectOffset((i as i64 + 1) * -8, Reg::Rbp),
                            ));
                        }
                        ObjType::Float => {
                            res.push(AsmInstruction::Movsd(
                                Address::Reg(Reg::Xmm0),
                                Address::IndirectOffset((i as i64 + 1) * -8, Reg::Rbp),
                            ));
                        }
                        ObjType::Array => todo!(),
                    }
                }
                if let Expr::Var { name } = callee.borrow() {
                    res.push(AsmInstruction::Call(name.to_string()));
                    let r_type = self.functions.get(&name.lexeme).unwrap();
                    return (res, *r_type);
                }
                todo!()
            }
        }
    }

    fn new_float_label(&mut self) -> String {
        self.float_count += 1;
        format!(".L.float.{}", self.float_count - 1)
    }

    fn new_str_label(&mut self) -> String {
        self.str_count += 1;
        format!(".L.str.{}", self.str_count - 1)
    }

    fn new_var_offset(&mut self) -> i64 {
        self.var_offset -= 8;
        self.var_offset - 8
    }

    fn get_label(&mut self) -> u64 {
        self.label_count += 1;
        self.label_count - 1
    }

    fn bin_op_fetch(&mut self, left: &Expr, right: &Expr) -> (Vec<AsmInstruction>, ObjType) {
        let ((left, l_type), (mut res, r_type)) = (self.expr(left), self.expr(right));
        if l_type != r_type {
            panic!(
                "The operands to a binary op must be the same type, got {:?}, {:?}",
                r_type, l_type
            );
        }
        if l_type == ObjType::Integer || l_type == ObjType::String {
            res.push(self.push());
            res.extend(left);
            res.push(self.pop(Reg::Rdi));
        } else if l_type == ObjType::Float {
            res.push(AsmInstruction::Movsd(
                Address::Reg(Reg::Xmm0),
                Address::IndirectOffset(-8, Reg::Rbp),
            ));
            res.extend(left);
            res.push(AsmInstruction::Movsd(
                Address::Reg(Reg::Xmm0),
                Address::IndirectOffset(-16, Reg::Rbp),
            ));
            res.push(AsmInstruction::Movsd(
                Address::IndirectOffset(-8, Reg::Rbp),
                Address::Reg(Reg::Xmm0),
            ));
            res.push(AsmInstruction::Movsd(
                Address::IndirectOffset(-16, Reg::Rbp),
                Address::Reg(Reg::Xmm1),
            ));
        }
        (res, l_type)
    }
}
