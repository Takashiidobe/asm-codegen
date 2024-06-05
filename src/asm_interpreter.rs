#![allow(unused)]
use std::{collections::HashMap, fmt};

use crate::token::ObjType;
use crate::{
    expr::Expr,
    stmt::Stmt,
    token::{Object, Token},
};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Codegen {
    depth: i64,
    instructions: Vec<AsmInstruction>,
    label_count: u64,
    anon_count: u64,
    float_count: u64,
    stack_offset: i64,
    pub vars: HashMap<String, (OffsetOrLabel, ObjType)>,
    pub functions: HashMap<Token, ObjType>,
    pub labels: HashMap<String, String>,
    strings: Vec<AsmInstruction>,
    floats: Vec<AsmInstruction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Reg {
    Rax,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    Rdx,
    Rcx,
    R8,
    R9,
    Al,
    Rip,
    Xmm0,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Address {
    Reg(Reg),
    Label(String),
    Immediate(i64),
    Indirect(Reg),
    IndirectOffset(i64, Reg),
    LabelOffset(String, Reg),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OffsetOrLabel {
    Offset(i64),
    Label(String),
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
    Add(Reg, Reg),
    Sub(Address, Reg),
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
            Movb(left, right) => f.write_fmt(format_args!("  movb {}, {}", left, right)),
            Movsd(left, right) => f.write_fmt(format_args!("  movsd {}, {}", left, right)),
            Movzb(left, right) => f.write_fmt(format_args!("  movzb {}, {}", left, right)),
            Cqo => f.write_fmt(format_args!("  cqo")),
            IMul(left, right) => f.write_fmt(format_args!("  imul {}, {}", left, right)),
            IDiv(left, right) => f.write_fmt(format_args!("  idiv {}, {}", left, right)),
            Add(left, right) => f.write_fmt(format_args!("  add {}, {}", left, right)),
            Sub(left, right) => f.write_fmt(format_args!("  sub {}, {}", left, right)),
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
        }
    }
}

impl Codegen {
    pub fn new() -> Self {
        Self {
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
        let size = self.vars.len();
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
            AsmInstruction::Label(".format_str".to_string()),
            AsmInstruction::Variable("string".to_string(), Some("\"%s\\n\"".to_string())),
            AsmInstruction::Label(".format_f64".to_string()),
            AsmInstruction::Variable("string".to_string(), Some("\"%f\\n\"".to_string())),
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

    fn offset(&mut self, obj: Option<Object>, obj_type: ObjType) -> OffsetOrLabel {
        todo!()
    }

    // make it so every expr returns a vector of results
    fn stmt(&mut self, stmt: &Stmt) -> Vec<AsmInstruction> {
        match stmt {
            Stmt::Expr { expr } => self.expr(expr).0,
            Stmt::Var { name, initializer } => todo!(),
            Stmt::Block { stmts } => todo!(),
            Stmt::Print { expr } => {
                let (mut expr_instruct, obj_type) = self.expr(expr);
                match obj_type {
                    ObjType::String => todo!(),
                    ObjType::Integer => todo!(),
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
                    ObjType::Bool => todo!(),
                    ObjType::Nil => todo!(),
                }
            }
            Stmt::Function {
                name,
                params,
                body,
                return_type,
            } => todo!(),
            Stmt::Return { keyword, value } => todo!(),
            Stmt::If { cond, then, r#else } => todo!(),
            Stmt::While { cond, body } => todo!(),
        }
    }

    // every expr should return a Vec<AsmInstruction>
    fn expr(&mut self, expr: &Expr) -> (Vec<AsmInstruction>, ObjType) {
        match expr {
            Expr::Binary { left, op, right } => todo!(),
            Expr::Assign { name, expr } => todo!(),
            Expr::Var { name } => todo!(),
            Expr::Logical { left, op, right } => todo!(),
            Expr::Unary { op, expr } => todo!(),
            Expr::Literal { value } => match value {
                Object::String(_) => todo!(),
                Object::Integer(_) => todo!(),
                Object::Float(f) => {
                    let label = self.get_float_label();
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
                Object::Bool(_) => todo!(),
                Object::Nil => todo!(),
            },
            Expr::Grouping { expr } => self.expr(expr),
            Expr::Call {
                callee,
                paren,
                arguments,
            } => todo!(),
        }
    }

    fn get_float_label(&mut self) -> String {
        self.float_count += 1;
        format!(".L.float.{}", self.float_count - 1)
    }

    fn get_anon_count(&mut self) -> u64 {
        self.anon_count += 1;
        self.anon_count - 1
    }

    fn get_count(&mut self) -> u64 {
        self.label_count += 1;
        self.label_count - 1
    }
}
