use std::process::Stdio;

use assert_cmd::cargo::CommandCargoExt as _;
use insta::glob;
use insta_cmd::{assert_cmd_snapshot, Command};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct TestOutput {
    status: i32,
    stdout: Vec<String>,
    stderr: Vec<String>,
}

#[test]
fn gen_bin() {
    glob!("input", "**/*.my", |path| {
        let mut cmd = Command::cargo_bin(env!("CARGO_PKG_NAME")).unwrap();

        let ps_child = cmd
            .arg(path)
            .stdout(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Could not create assembly");

        let out_name = format!("{}.out", path.display());

        Command::new("cc")
            .arg("-o")
            .arg(&out_name)
            .arg("-xassembler")
            .arg("-")
            .stdin(Stdio::from(ps_child.stdout.unwrap()))
            .spawn()
            .expect("Could not assemble binary");

        assert_cmd_snapshot!(Command::new(&out_name));
    });
}
