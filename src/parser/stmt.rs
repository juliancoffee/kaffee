use super::{spanned, Spanned};
use chumsky::prelude::*;

#[cfg(test)]
use super::pattern::Pattern;
#[cfg(test)]
use super::Name;

use super::{
    bind::{bind, Bind},
    ty::{typedef, TypeDef},
};

use super::{
    expr::{expression, Expr},
    lex::Token,
    SubParser,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Let {
        bind: Spanned<Bind>,
        expr: Spanned<Expr>,
    },
    TypeDef(TypeDef),
    ModuleStruct(String, Vec<Spanned<Self>>),
    Commented(Spanned<String>, Box<Spanned<Self>>),
}

impl Statement {
    #[cfg(test)]
    pub(super) fn let_bound(pat: Pattern, expr: Expr) -> Self {
        Self::Let {
            bind: Bind::Bound(pat),
            expr,
        }
    }

    #[cfg(test)]
    pub(super) fn let_abstraction(
        name: Name,
        args: Vec<Pattern>,
        expr: Expr,
    ) -> Self {
        Self::Let {
            bind: Bind::Abstraction(name, args),
            expr,
        }
    }

    #[cfg(test)]
    pub(super) fn module_struct(name: &str, defs: Vec<Self>) -> Self {
        Self::ModuleStruct(name.to_owned(), defs)
    }

    pub(super) fn with_doc(
        docstring: Spanned<String>,
        stmt: Spanned<Self>,
    ) -> Self {
        Self::Commented(docstring, Box::new(stmt))
    }
}

fn let_stmt() -> impl SubParser<Spanned<Statement>> {
    just(Token::Let)
        .ignore_then(bind())
        .then_ignore(just(Token::Is))
        .then(expression())
        .then_ignore(just(Token::Semicolon))
        .map(|(bind, expr)| Statement::Let { bind, expr })
        .map_with_span(spanned)
}

fn module<S: SubParser<Spanned<Statement>>>(
    stmt: S,
) -> impl SubParser<Spanned<Statement>> {
    just(Token::Module)
        .ignore_then(select! {Token::Ident(i) => i})
        .then_ignore(just(Token::Is))
        .then_ignore(just(Token::Struct))
        .then(stmt.repeated())
        .then_ignore(just(Token::End))
        .map(|(name, defs)| Statement::ModuleStruct(name, defs))
        .map_with_span(spanned)
}

fn commented<S: SubParser<Spanned<Statement>>>(
    stmt: S,
) -> impl SubParser<Spanned<Statement>> {
    select! {Token::Doc(doc) => doc}
        .map_with_span(spanned)
        .then(stmt)
        .map(|(doc, stmt)| Statement::with_doc(doc, stmt))
        .map_with_span(spanned)
}

pub(super) fn statement() -> impl SubParser<Spanned<Statement>> {
    recursive(|stmt| {
        let let_stmt = let_stmt();
        let module = module(stmt.clone());
        let commented = commented(stmt);
        let typedef = typedef().map(Statement::TypeDef).map_with_span(spanned);

        choice((commented, let_stmt, module, typedef))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{
        expr::{BinOp, UnOp},
        tests::tokens,
    };

    fn parse_stmt(src: &str) -> Result<Statement, Vec<Simple<Token>>> {
        let token_stream = tokens(src);

        statement().parse(token_stream)
    }

    #[test]
    fn parse_let_stmt() {
        let src = "let x = 50;";
        assert_eq!(
            parse_stmt(src),
            Ok(Statement::let_bound(Pattern::ident("x"), Expr::Int(50)))
        );
    }

    #[test]
    fn parse_let_unit_stmt() {
        let src = r#"
let () = print "wow";
        "#;

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::let_bound(
                Pattern::Tuple(vec![]),
                Expr::app(Expr::ident("print"), Expr::string("wow")),
            ))
        );
    }

    #[test]
    fn parse_let_discard_stmt() {
        let src = r#"
let _ = rm "tmp.kf";
        "#;

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::let_bound(
                Pattern::Wildcard,
                Expr::app(Expr::ident("rm"), Expr::string("tmp.kf")),
            ))
        );
    }

    #[test]
    fn parse_let_abstraction_stmt() {
        let src = "let neg x = - x;";

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::let_abstraction(
                Name::plain("neg"),
                vec![Pattern::ident("x")],
                Expr::unary_with(UnOp::Negative, Expr::ident("x"))
            ))
        );
    }

    #[test]
    fn parse_module_stmt() {
        let src = r#"
module Ops = struct
    let add x y = x + y;
    let sub x y = x - y;
end
        "#;

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::module_struct(
                "Ops",
                vec![
                    Statement::let_abstraction(
                        Name::plain("add"),
                        vec![Pattern::ident("x"), Pattern::ident("y")],
                        Expr::binop_with(
                            BinOp::Sum,
                            Expr::ident("x"),
                            Expr::ident("y"),
                        )
                    ),
                    Statement::let_abstraction(
                        Name::plain("sub"),
                        vec![Pattern::ident("x"), Pattern::ident("y")],
                        Expr::binop_with(
                            BinOp::Sub,
                            Expr::ident("x"),
                            Expr::ident("y"),
                        )
                    ),
                ]
            ))
        );
    }

    #[test]
    fn parse_commented_stmt() {
        let src = r#"
// (+) operator wrapper
let add x y = x + y;
        "#;

        assert_eq!(
            parse_stmt(src),
            Ok(Statement::with_doc(
                "(+) operator wrapper".to_owned(),
                Statement::let_abstraction(
                    Name::plain("add"),
                    vec![Pattern::ident("x"), Pattern::ident("y")],
                    Expr::binop_with(
                        BinOp::Sum,
                        Expr::ident("x"),
                        Expr::ident("y"),
                    )
                ),
            ))
        );
    }
}
