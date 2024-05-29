from antlr4 import ParserRuleContext

from project.gql.generated.project.gql.gqlListener import gqlListener


class ListenerWithRes(gqlListener):
    def __init__(self, res):
        super(gqlListener, self).__init__()
        self.res = None


class NodeCountListener(ListenerWithRes):
    def __init__(self) -> None:
        super(ListenerWithRes, self).__init__()
        self.res = 0

    def enterEveryRule(self, _):
        self.res += 1


class ProgBuildListener(ListenerWithRes):
    def __init__(self):
        super(ListenerWithRes, self).__init__()
        self.res = ""

    def enterEveryRule(self, ruleCtx):
        self.res += ruleCtx.getText()


def exec_listener(listener: ListenerWithRes, tree: ParserRuleContext) -> any:
    tree.enterRule(listener)

    return listener.res
