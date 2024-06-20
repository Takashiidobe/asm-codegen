# Asm-codegen

This project is a small language that compiles to x86_64 linux assembly.

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
