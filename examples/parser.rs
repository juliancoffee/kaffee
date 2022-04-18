use kaffee::parser;
use std::io::Read;

fn main() -> std::io::Result<()> {
    let filename = &std::env::args().collect::<Vec<_>>()[1];
    let mut file = std::fs::File::open(filename)?;
    let mut src = String::new();
    file.read_to_string(&mut src)?;

    let ast = parser::Ast::parse_expect(&src);
    println!("{ast:#?}");

    Ok(())
}
