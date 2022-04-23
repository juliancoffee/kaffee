use chumsky::prelude::*;

use super::{
    grouping,
    lex::{Op, Token},
    spanned, tuple_like, Name, Spanned, SubParser,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Ident(String),
    Var(String),
    Product(Vec<Spanned<Self>>),
    App(Box<Spanned<Self>>, Vec<Spanned<Self>>),
}

impl Type {
    #[cfg(test)]
    pub(super) fn ident(name: &str) -> Self {
        Self::Ident(name.to_owned())
    }

    #[cfg(test)]
    pub(super) fn var(name: &str) -> Self {
        Self::Var(name.to_owned())
    }

    pub(super) fn app(
        constructor: Spanned<Self>,
        args: Vec<Spanned<Self>>,
    ) -> Self {
        Self::App(Box::new(constructor), args)
    }
}

fn product_type<P: SubParser<Spanned<Type>> + Clone>(
    type_expr: P,
) -> impl SubParser<Spanned<Type>> + Clone {
    type_expr
        .separated_by(just(Token::Op(Op::Product)))
        .at_least(2)
        .collect::<Vec<_>>()
        .map(Type::Product)
        .map_with_span(spanned)
}

fn type_app<P: SubParser<Spanned<Type>> + Clone + 'static>(
    type_expr: P,
) -> impl SubParser<Spanned<Type>> + Clone {
    recursive(|app| {
        let atom = app.or(type_expr.clone());
        let args = choice((
            tuple_like(atom.clone(), std::convert::identity),
            atom.clone().map(|t| vec![t]),
        ));

        type_expr
            .then(args)
            .map(|(cons, args)| Type::app(cons, args))
            .map_with_span(spanned)
    })
}

fn ty() -> impl SubParser<Spanned<Type>> {
    recursive(|type_expr| {
        let ident = select! {
            Token::Ident(i) => Type::Ident(i),
            Token::TypeParameter(p) => Type::Var(p),
        };
        let ident = ident.map_with_span(spanned);
        let grouping = grouping(type_expr.clone());

        let atom = ident.or(grouping);

        // Type application ('a constructor)
        let app = type_app(atom.clone());
        // Type product ('a * 'b)
        let product = product_type(app.clone().or(atom.clone()));

        product.or(app).or(atom)
    })
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeBind {
    Bound(Spanned<Name>),
    // TODO: make abstraction use Name as well to be able to create spells for
    // list literals.
    Abstraction(Spanned<Name>, Vec<Spanned<String>>),
}

impl TypeBind {
    #[cfg(test)]
    pub(super) fn bound(name: Name) -> Self {
        Self::Bound(name)
    }

    #[cfg(test)]
    pub(super) fn abstraction(name: Name, vars: Vec<&str>) -> Self {
        Self::Abstraction(name, vars.into_iter().map(str::to_owned).collect())
    }
}

fn type_bind() -> impl SubParser<TypeBind> {
    let one_vec = |x| vec![x];
    let name = select! {
        Token::Ident(i) => Name::Plain(i),
        Token::LiteralSpell(s) => Name::Spell(s),
    };
    let name = name.map_with_span(spanned);

    let bound = name.map(TypeBind::Bound);

    let type_var = select! {
        Token::TypeParameter(p) => p,
    };
    let type_arg = choice((
        tuple_like(type_var.map_with_span(spanned), std::convert::identity),
        type_var.map_with_span(spanned).map(one_vec),
    ));
    let abstraction = name
        .then(type_arg)
        .map(|(name, vars)| TypeBind::Abstraction(name, vars));

    abstraction.or(bound)
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeDef {
    Alias(TypeBind, Spanned<Type>),
}

#[allow(clippy::let_and_return)]
pub(super) fn typedef() -> impl SubParser<TypeDef> {
    let alias = just(Token::Type)
        .ignore_then(type_bind())
        .then_ignore(just(Token::Is))
        .then(ty())
        .then_ignore(just(Token::Semicolon))
        .map(|(bind, ty)| TypeDef::Alias(bind, ty));

    alias
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::tests::tokens;

    fn parse_typedef(src: &str) -> Result<TypeDef, Vec<Simple<Token>>> {
        let token_stream = tokens(src);

        typedef().parse(token_stream)
    }

    #[test]
    fn parse_type_alias_stmt() {
        let src = "type idx = nat;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::bound(Name::plain("idx")),
                Type::ident("nat"),
            ))
        );
    }

    #[test]
    fn parse_type_alias_tuple_stmt() {
        let src = "type point = int * int;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::bound(Name::plain("point")),
                Type::Product(vec![Type::ident("int"), Type::ident("int")]),
            ))
        );
    }

    #[test]
    fn parse_type_alias_with_spell_stmt() {
        let src = "type @integer = int64;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::bound(Name::spell("integer")),
                Type::ident("int64"),
            ))
        );
    }

    #[test]
    fn parse_type_alias_trivial_generic() {
        let src = "type just 'a = 'a;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::abstraction(Name::plain("just"), vec!["a"]),
                Type::var("a"),
            ))
        );
    }

    #[test]
    fn parse_type_alias_generic_tuple_stmt() {
        let src = "type pair ('a, 'b) = 'a * 'b;";
        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::abstraction(Name::plain("pair"), vec!["a", "b"]),
                Type::Product(vec![Type::var("a"), Type::var("b")]),
            ))
        );
    }

    #[test]
    fn parse_type_alias_generic_triple_stmt() {
        let src = "type triple ('a, 'b, 'c) = ('a * 'b) * 'c;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::abstraction(
                    Name::plain("triple"),
                    vec!["a", "b", "c"]
                ),
                Type::Product(vec![
                    Type::Product(vec![Type::var("a"), Type::var("b")]),
                    Type::var("c"),
                ]),
            ))
        );
    }

    #[test]
    fn parse_type_alias_type_app_expr() {
        let src = "type ints = list int;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::Bound(Name::plain("ints")),
                Type::app(Type::ident("list"), vec![Type::ident("int")]),
            ))
        );
    }

    #[test]
    fn parse_type_alias_type_complex() {
        let src = "type call_results = list result (int, string);";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::Bound(Name::plain("call_results")),
                Type::app(
                    Type::ident("list"),
                    vec![Type::app(
                        Type::ident("result"),
                        vec![Type::ident("int"), Type::ident("string")]
                    )]
                ),
            ))
        );
    }
}
