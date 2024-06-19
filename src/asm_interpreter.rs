#![allow(unused)]
use std::borrow::Borrow as _;
use std::{collections::HashMap, fmt};

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
    pub functions: HashMap<Token, ObjType>,
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
        })
    }
}

const ARG_REGS: [Reg; 6] = [Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9];

#[derive(Debug, Clone, PartialEq)]
pub enum AsmInstruction {
    Section(String),
    Variable(String, Option<String>),
    Label(String),
    Xor(Reg, Reg),
    Push(Reg),
    Pop(Reg),
    Call(String),
    Lea(Address, Address),
    Mov(Address, Address),
    Movq(Address, Address),
    Movb(Address, Address),
    Movsd(Address, Address),
    Movzb(Address, Address),
    Sete(Reg),
    Setne(Reg),
    Setl(Reg),
    Setle(Reg),
    Setg(Reg),
    Setge(Reg),
    Cqo,
    Ret,
    Test(Reg, Reg),
    Cmp(Address, Reg),
    IMul(Reg, Reg),
    IDiv(Reg, Reg),
    Add(Address, Reg),
    Addsd(Reg, Reg),
    Sub(Address, Reg),
    Subsd(Reg, Reg),
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
            Xor(left, right) => f.write_fmt(format_args!("  xor {}, {}", left, right)),
            Push(reg) => f.write_fmt(format_args!("  push {}", reg)),
            Pop(reg) => f.write_fmt(format_args!("  pop {}", reg)),
            Sete(reg) => f.write_fmt(format_args!("  sete {}", reg)),
            Setne(reg) => f.write_fmt(format_args!("  setne {}", reg)),
            Setl(reg) => f.write_fmt(format_args!("  setl {}", reg)),
            Setle(reg) => f.write_fmt(format_args!("  setle {}", reg)),
            Setg(reg) => f.write_fmt(format_args!("  setg {}", reg)),
            Setge(reg) => f.write_fmt(format_args!("  setge {}", reg)),
            Ret => f.write_fmt(format_args!("  ret")),
            Call(fn_name) => f.write_fmt(format_args!("  call {}", fn_name)),
            Test(left, right) => f.write_fmt(format_args!("  test {}, {}", left, right)),
            Cmp(left, right) => f.write_fmt(format_args!("  cmp {}, {}", left, right)),
            Mov(left, right) => f.write_fmt(format_args!("  mov {}, {}", left, right)),
            Movq(left, right) => f.write_fmt(format_args!("  movq {}, {}", left, right)),
            Movb(left, right) => f.write_fmt(format_args!("  movb {}, {}", left, right)),
            Movsd(left, right) => f.write_fmt(format_args!("  movsd {}, {}", left, right)),
            Movzb(left, right) => f.write_fmt(format_args!("  movzb {}, {}", left, right)),
            Cqo => f.write_fmt(format_args!("  cqo")),
            IMul(left, right) => f.write_fmt(format_args!("  imul {}, {}", left, right)),
            IDiv(left, right) => f.write_fmt(format_args!("  idiv {}, {}", left, right)),
            Add(left, right) => f.write_fmt(format_args!("  add {}, {}", left, right)),
            Addsd(left, right) => f.write_fmt(format_args!("  addsd {}, {}", left, right)),
            Sub(left, right) => f.write_fmt(format_args!("  sub {}, {}", left, right)),
            Subsd(left, right) => f.write_fmt(format_args!("  subsd {}, {}", left, right)),
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
        let size = self.vars.len() + 33;
        let mut program = vec![AsmInstruction::Variable("text".to_string(), None)];
        let prologue = self.prologue(size);
        let epilogue = self.epilogue();

        program.extend(functions);
        program.extend(prologue);
        program.extend(body);
        program.extend(epilogue);
        if !self.strings.is_empty() {
            program.push(AsmInstruction::Variable("data".to_string(), None));
            program.extend(self.strings.clone());
        }
        if !self.floats.is_empty() {
            program.push(AsmInstruction::Variable("data".to_string(), None));
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
            AsmInstruction::Variable("globl".to_string(), Some("main".to_string())),
            AsmInstruction::Label("main".to_string()),
            AsmInstruction::Push(Reg::Rbp),
            AsmInstruction::Mov(Address::Reg(Reg::Rsp), Address::Reg(Reg::Rbp)),
            AsmInstruction::Sub(
                Address::Immediate(self.align_to(size * 8, 16) as i64),
                Reg::Rsp,
            ),
        ]
    }

    fn epilogue(&mut self) -> Vec<AsmInstruction> {
        vec![
            AsmInstruction::Xor(Reg::Rax, Reg::Rax),
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
                    res.push(AsmInstruction::Xor(Reg::Rax, Reg::Rax));
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
                            AsmInstruction::Movb(Address::Immediate(0), Address::Reg(Reg::Al)),
                            AsmInstruction::Call("printf".to_string()),
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
                }
            }
            Stmt::Function {
                name,
                params,
                body,
                return_type,
            } => todo!(),
            Stmt::Return { keyword, value } => {
                let mut instructions = vec![];

                instructions.extend(if let Some(val) = value {
                    self.expr(val).0
                } else {
                    self.expr(&Expr::Literal { value: Object::Nil }).0
                });
                instructions.extend(vec![
                    AsmInstruction::Xor(Reg::Rax, Reg::Rax),
                    AsmInstruction::Leave,
                    AsmInstruction::Ret,
                ]);
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
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    if r_type == ObjType::Integer {
                        res.push(AsmInstruction::Add(Address::Reg(Reg::Rdi), Reg::Rax));
                    }
                    if r_type == ObjType::Float {
                        res.push(AsmInstruction::Addsd(Reg::Xmm1, Reg::Xmm0));
                    }
                    (res, r_type)
                }
                Token {
                    r#type: TokenType::Minus,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    if r_type == ObjType::Integer {
                        res.push(AsmInstruction::Sub(Address::Reg(Reg::Rdi), Reg::Rax));
                    }
                    if r_type == ObjType::Float {
                        res.push(AsmInstruction::Subsd(Reg::Xmm1, Reg::Xmm0));
                    }
                    (res, r_type)
                }

                Token {
                    r#type: TokenType::EqualEqual,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                    res.push(AsmInstruction::Sete(Reg::Al));
                    res.push(AsmInstruction::Movzb(
                        Address::Reg(Reg::Al),
                        Address::Reg(Reg::Rax),
                    ));
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::BangEqual,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                    res.push(AsmInstruction::Setne(Reg::Al));
                    res.push(AsmInstruction::Movzb(
                        Address::Reg(Reg::Al),
                        Address::Reg(Reg::Rax),
                    ));
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::Less,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                    res.push(AsmInstruction::Setl(Reg::Al));
                    res.push(AsmInstruction::Movzb(
                        Address::Reg(Reg::Al),
                        Address::Reg(Reg::Rax),
                    ));
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::LessEqual,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                    res.push(AsmInstruction::Setle(Reg::Al));
                    res.push(AsmInstruction::Movzb(
                        Address::Reg(Reg::Al),
                        Address::Reg(Reg::Rax),
                    ));
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::Greater,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                    res.push(AsmInstruction::Setg(Reg::Al));
                    res.push(AsmInstruction::Movzb(
                        Address::Reg(Reg::Al),
                        Address::Reg(Reg::Rax),
                    ));
                    (res, ObjType::Bool)
                }
                Token {
                    r#type: TokenType::GreaterEqual,
                    ..
                } => {
                    let (mut res, r_type) = self.bin_op_fetch(left, right);
                    res.push(AsmInstruction::Cmp(Address::Reg(Reg::Rdi), Reg::Rax));
                    res.push(AsmInstruction::Setge(Reg::Al));
                    res.push(AsmInstruction::Movzb(
                        Address::Reg(Reg::Al),
                        Address::Reg(Reg::Rax),
                    ));
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
                        res.push(AsmInstruction::Xor(Reg::Rax, Reg::Rax));
                    }

                    self.vars.insert(name.lexeme.clone(), (*off, r_type));
                } else {
                    panic!("Var {name} was not defined");
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
            Expr::Logical { left, op, right } => todo!(),
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
                    // each float needs to have its own unique location in memory.
                    // ObjType float should set one up.
                    // so we need a function that creates a new stack var and assigns it
                    // and when we load it, we can use that as well.
                    (
                        vec![AsmInstruction::Movsd(
                            Address::LabelOffset(label, Reg::Rip),
                            Address::Reg(Reg::Xmm0),
                        )],
                        ObjType::Float,
                    )
                }
                Object::Identifier(_) => todo!(),
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
                callee,
                paren,
                arguments,
            } => todo!(),
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
        let ((mut res, r_type), (left, l_type)) = (self.expr(right), self.expr(left));
        let res_copy = res.clone();
        if l_type != r_type {
            panic!(
                "The operands to a binary op must be the same type, got {:?}, {:?}",
                r_type, l_type
            );
        }
        if l_type == ObjType::Integer {
            res.push(self.push());
            res.extend(left);
            res.push(self.pop(Reg::Rdi));
        } else if l_type == ObjType::Float {
            res.push(AsmInstruction::Movsd(
                Address::Reg(Reg::Xmm0),
                // TODO: this needs to have the offset of the float in the stack
                Address::IndirectOffset(-8, Reg::Rbp),
            ));
            res.extend(left);
            res.push(AsmInstruction::Movsd(
                // TODO: this needs to have the offset of the float in the stack
                Address::IndirectOffset(-8, Reg::Rbp),
                Address::Reg(Reg::Xmm1),
            ));
        } else if l_type == ObjType::String {
            res.push(AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::Reg(Reg::Rdi),
            ));
            res.push(AsmInstruction::Call("strlen".to_string()));
            res.push(AsmInstruction::Movq(
                Address::Reg(Reg::Rax),
                Address::Reg(Reg::Rbx),
            ));
            res.extend(left.clone());
            res.push(AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::Reg(Reg::Rdi),
            ));
            res.push(AsmInstruction::Call("strlen".to_string()));
            res.push(AsmInstruction::Add(Address::Reg(Reg::Rbx), Reg::Rax));
            res.push(AsmInstruction::Add(Address::Immediate(1), Reg::Rax));
            res.push(AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::Reg(Reg::Rdi),
            ));
            res.push(AsmInstruction::Call("malloc".to_string()));
            res.push(AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::IndirectOffset(-8, Reg::Rbp),
            ));
            res.push(AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::Reg(Reg::Rdi),
            ));
            res.extend(res_copy);
            res.push(AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::Reg(Reg::Rsi),
            ));
            res.push(AsmInstruction::Call("strcpy".to_string()));
            res.extend(left);
            res.push(AsmInstruction::Mov(
                Address::Reg(Reg::Rax),
                Address::Reg(Reg::Rsi),
            ));
            res.push(AsmInstruction::Call("strcat".to_string()));
        }
        (res, l_type)
    }
}
