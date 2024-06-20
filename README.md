# Asm-codegen

This project is a small language that compiles to x86_64 linux assembly.

## Example

Take this program that iteratively calculates the 10th fibonacci number.

```sh
$ cat tests/input/programs/ex.my
fun fib(n: i64) -> i64 {
	var prevprev = 0;
	var prev = 0;
	var curr = 1;

	var i = 1;

	while (i < n) {
		prevprev = prev;
		prev = curr;
		curr = prevprev + prev;
		i = i + 1;
	}

	return curr;
}

print fib(10);
```

We can generate the assembly for the program:

```sh
$ cargo r -q -- tests/input/programs/ex.my > ex.s
```

And check it out:

```sh
$ cat ex.s
```

```asm
  .text
fib:
  mov $0, %rax
  mov %rax, -272(%rbp)
  xor %rax, %rax
  mov $0, %rax
  mov %rax, -280(%rbp)
  xor %rax, %rax
  mov $1, %rax
  mov %rax, -288(%rbp)
  xor %rax, %rax
  mov $1, %rax
  mov %rax, -296(%rbp)
  xor %rax, %rax
.L.begin.0:
  mov -8(%rbp), %rax
  push %rax
  mov -296(%rbp), %rax
  pop %rdi
  cmp %rdi, %rax
  setl %al
  movzb %al, %rax
  cmp $0, %rax
  je .L.end.0
  mov -280(%rbp), %rax
  mov %rax, -272(%rbp)
  xor %rax, %rax
  mov -288(%rbp), %rax
  mov %rax, -280(%rbp)
  xor %rax, %rax
  mov -280(%rbp), %rax
  push %rax
  mov -272(%rbp), %rax
  pop %rdi
  add %rdi, %rax
  mov %rax, -288(%rbp)
  xor %rax, %rax
  mov $1, %rax
  push %rax
  mov -296(%rbp), %rax
  pop %rdi
  add %rdi, %rax
  mov %rax, -296(%rbp)
  xor %rax, %rax
  jmp .L.begin.0
.L.end.0:
  mov -288(%rbp), %rax
  ret
  ret
  .globl main
main:
  push %rbp
  mov %rsp, %rbp
  sub $304, %rsp
  mov $10, %rax
  mov %rax, -8(%rbp)
  call fib
  mov %rax, %rsi
  mov $.format_i64, %rdi
  movb $0, %al
  call printf
  xor %rax, %rax
  leave
  ret
  .section .rodata
.format_i64:
  .string "%d\n"
.format_f64:
  .string "%f\n"
.nil_string:
  .string "nil"
.true_string:
  .string "true"
.false_string:
  .string "false"
neg_mask:
  .quad 0x8000000000000000
```

And assemble it, which returns the following:

```sh
$ cc ex.s
$ ./a.out
55
```

## Features

### Variables

```
var x = 0;
x = 10;
print x; // print 10
var y = "hi";
y = "world";
print y; // print "world"
```

### If-else

Ifs and elses are implemented:

```
var x = 1;
var y = 2;

if (x == 1) {
  if (y == 2) {
    print "x == 1 && y == 2"; // print this
  } else {
    print "x == 1 && y != 2";
  }
} else {
  print "x != 1";
}
```

### Functions

Functions look like this, with typing. Currently the language doesn't do
anything with the types.

```
fun add(a: i64, b: i64) -> i64 {
  if (a == 0) { return b; }
  return 1 + add(a - 1, b);
}

print add(3, 2); // prints 5
```

### Arrays

Arrays are supported:

```
var x = [0 + 1, 1 + 1, 1 + 2]; // x is [1,2,3]
```

There's no array access operator, and array printing isn't implemented
yet.

## TODO

- Float operations don't work, since loading float registers isn't done
  properly.
- Variables need to be scoped properly.
- Recursion is broken.
- String concatenation doesn't work (concatenating more than 2 strings segfaults since it doesn't allocate a new string on the heap)
- Maybe Hindley Milner types?

## Development

### Architecture

The main files in the repository are the tree-walking interpreter:
[Interpreter](./src/asm_interpreter.rs) which generates the assembly
file that can be compiled by an assembler, the parser, which is a
recursive descent parser that parses the language itself:
[Parser](./src/parser.rs), and the lexer: [Lexer](./src/lexer.rs) which
tokenizes the language. There are Statements: [Stmt](./src/stmt.rs) and
Expressions as well: [Expr](./src/expr.rs) as well as tokens
[Token](./src/token.rs), which are returned by the lexer.

### Functions

Currently, the language supports up to 32 parameters passed to a
function. These 32 slots are reserved on the stack, and every function
can use those stack spaces, before moving its return value into %rax.
Thus, functions don't use registers for fast access. This can be
optimized later in a pass.

### Testing

To test the project, `asm-codegen` uses insta, which is a snapshot
testing library for Rust. This saves the output of tests (which are
located in `tests/input/**/*.my` into files, so they can be manually
reviewed for correctness.

Any new feature should have corresponding tests.

### Fuzzing

In the future, I'd like to use [bnf-gen](https://baturin.org/tools/bnfgen/)
to generate random programs following the bnf of the language, and use
that to test for bugs.
