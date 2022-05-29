use chumsky::prelude::*;

use super::{
    common::{from_module, grouping, tuple_like},
    lex::{Op, Token},
    spanned, Name, Spanned, SubParser,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Ident(String),
    Var(String),
    FromModule(Spanned<String>, Box<Spanned<Self>>),
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

    pub(super) fn from_module(
        mod_name: Spanned<String>,
        t: Spanned<Self>,
    ) -> Self {
        Self::FromModule(mod_name, Box::new(t))
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

fn imported<P: SubParser<Spanned<Type>> + Clone + 'static>(
    type_expr: P,
) -> impl SubParser<Spanned<Type>> + Clone {
    from_module(type_expr, Type::from_module)
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

        // Type application (constructor 'a)
        let app = type_app(atom.clone());
        let complex = app.or(atom.clone());
        // Type product ('a * 'b)
        let product = product_type(complex.clone());
        let complex = product.or(complex);
        // Type from module (Mod.t)
        let from_module = imported(complex.clone());
        let complex = from_module.or(complex);

        complex.or(atom)
    })
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeBind {
    Bound(Spanned<Name>),
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
pub enum Variant {
    Case(Spanned<String>, Spanned<Type>),
    Tag(Spanned<String>),
}

impl Variant {
    #[cfg(test)]
    fn tag(name: &str) -> Self {
        Self::Tag(name.to_owned())
    }

    #[cfg(test)]
    fn case(name: &str, def: Type) -> Self {
        Self::Case(name.to_owned(), def)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeDef {
    Alias(TypeBind, Spanned<Type>),
    Enum(TypeBind, Vec<Variant>),
}

fn alias() -> impl SubParser<TypeDef> {
    just(Token::Type)
        .ignore_then(type_bind())
        .then_ignore(just(Token::Is))
        .then(ty())
        .then_ignore(just(Token::Semicolon))
        .map(|(bind, ty)| TypeDef::Alias(bind, ty))
}

fn variant() -> impl SubParser<Variant> {
    let tag = select! {
        Token::Ident(i) => i,
    }
    .map_with_span(spanned);

    choice((
        tag.then_ignore(just(Token::Of))
            .then(ty())
            .map(|(tag, def)| Variant::Case(tag, def)),
        tag.map(Variant::Tag),
    ))
}

fn variants() -> impl SubParser<TypeDef> {
    just(Token::Type)
        .ignore_then(type_bind())
        .then_ignore(just(Token::Is))
        .then(variant().separated_by(just(Token::Either)))
        .then_ignore(just(Token::Semicolon))
        .map(|(bind, variants)| TypeDef::Enum(bind, variants))
}

#[allow(clippy::let_and_return)]
pub(super) fn typedef() -> impl SubParser<TypeDef> {
    choice((alias(), variants()))
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

    #[test]
    fn parse_type_enum_tags() {
        let src = "type colour = Black | White;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Enum(
                TypeBind::Bound(Name::plain("colour")),
                vec![Variant::tag("Black"), Variant::tag("White")]
            ))
        );
    }

    #[test]
    fn parse_type_enum() {
        let src = "type id = Num of nat | Name of string;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Enum(
                TypeBind::Bound(Name::plain("id")),
                vec![
                    Variant::case("Num", Type::ident("nat")),
                    Variant::case("Name", Type::ident("string")),
                ]
            ))
        );
    }

    #[test]
    fn parse_type_option() {
        let src = "type option 'a = Some of 'a | None;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Enum(
                TypeBind::abstraction(Name::plain("option"), vec!["a"]),
                vec![
                    Variant::case("Some", Type::var("a")),
                    Variant::tag("None"),
                ]
            ))
        );
    }

    #[test]
    fn parse_type_from_module() {
        let src = "type @integer = Int64Internal.t;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::Bound(Name::spell("integer")),
                Type::from_module("Int64Internal".to_owned(), Type::ident("t")),
            ))
        );
    }

    #[test]
    fn parse_type_nested_module() {
        let src = "type target = X.Y.t;";

        assert_eq!(
            parse_typedef(src),
            Ok(TypeDef::Alias(
                TypeBind::Bound(Name::plain("target")),
                Type::from_module(
                    "X".to_owned(),
                    Type::from_module("Y".to_owned(), Type::ident("t")),
                ),
            ))
        );
    }
}
