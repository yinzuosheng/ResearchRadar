from web.app import WORKBENCH_NAV_ITEMS, render_sidebar


class Sidebar:
    def __init__(self):
        self.markdown_calls = []
        self.radio_calls = []
        self.button_calls = []

    def markdown(self, value):
        self.markdown_calls.append(value)

    def radio(self, label, options, **kwargs):
        self.radio_calls.append((label, list(options), kwargs))
        return options[0]

    def button(self, label, **kwargs):
        self.button_calls.append((label, kwargs))
        return False


class FakeStreamlit:
    def __init__(self):
        self.sidebar = Sidebar()
        self.session_state = {}


def test_workbench_sidebar_has_grouped_research_workflows():
    grouped = {}
    for group, label in WORKBENCH_NAV_ITEMS:
        grouped.setdefault(group, []).append(label)

    assert grouped == {
        "工作台": ["科研助手", "知识库", "文献库", "研究方案"],
        "维护": ["知识库维护", "任务与日志", "数据源设置"],
    }


def test_removed_experimental_tools_are_not_exposed_in_sidebar():
    labels = [label for _group, label in WORKBENCH_NAV_ITEMS]
    assert "趋势报告" not in labels
    assert "论文对比" not in labels


def test_render_sidebar_defaults_to_research_agent_and_renders_group_headers():
    st = FakeStreamlit()

    selected = render_sidebar(st)

    assert selected == "科研助手"
    assert st.sidebar.radio_calls == []
    options = [label for label, _kwargs in st.sidebar.button_calls]
    assert any("科研助手" in value for value in st.sidebar.markdown_calls + options)
    assert all("●" not in label and "○" not in label for label in options)
    assert any("workbench-divider" in value for value in st.sidebar.markdown_calls)
    assert any("维护" in value for value in st.sidebar.markdown_calls)
    assert all("⌂" not in label and "▱" not in label for label in options)


def test_render_sidebar_recovers_from_removed_module_in_session_state():
    st = FakeStreamlit()
    st.session_state["research_workbench_module"] = "趋势报告"

    selected = render_sidebar(st)

    assert selected == "科研助手"
    assert st.session_state["research_workbench_module"] == "科研助手"
