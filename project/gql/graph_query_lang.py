from antlr4 import ParserRuleContext, CommonTokenStream, InputStream

from project.gql.generated.project.gql.gqlLexer import gqlLexer
from project.gql.generated.project.gql.gqlParser import gqlParser
from project.gql.listeners import NodeCountListener, ProgBuildListener, exec_listener


def prog_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    parser = gqlParser(CommonTokenStream(gqlLexer(InputStream(program))))

    return parser.prog(), parser.getNumberOfSyntaxErrors() == 0


def nodes_count(tree: ParserRuleContext) -> int:
    return exec_listener(NodeCountListener(), tree)


def tree_to_prog(tree: ParserRuleContext) -> str:
    return exec_listener(ProgBuildListener(), tree)
