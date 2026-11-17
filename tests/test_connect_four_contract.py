"""ConnectFour 的“行为契约”测试。

这组用例不是凭感觉写的：每一条都对应 killrate 在
`envs/connect_four.py` 上找到的一个存活变异体（把代码改错、现有测试却仍然全绿）。
目标是把这些“没人断言过的行为”固定下来，而不是堆覆盖率。

对应关系见每条用例的注释（M 编号来自 runs/connect4.md）。
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.connect_four import ConnectFour

# 6x7 的平局终局：任意方向都没有四连（横/竖最多 2 连，两条对角线最多 2 连）
DRAW_BOARD = [
    [1, 1, 2, 2, 1, 1, 2],
    [2, 2, 1, 1, 2, 2, 1],
    [1, 1, 2, 2, 1, 1, 2],
    [2, 2, 1, 1, 2, 2, 1],
    [1, 1, 2, 2, 1, 1, 2],
    [2, 2, 1, 1, 2, 2, 1],
]


def play(env: ConnectFour, moves: list[int]) -> None:
    for col in moves:
        env.step(col)


def test_new_and_reset_state_is_fully_initialized():
    """M011 / M013 / M016：形状、winner、terminated 在构造与 reset 后都要归位。"""
    env = ConnectFour()
    assert env.observation_shape == (2, 6, 7)
    assert env.winner == 0
    assert env.terminated is False
    assert env.current_player == 1  # M012：构造函数里的初始行棋方

    obs = env.reset()  # M018：reset 的返回值以前没人接
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2, 6, 7)
    assert obs.sum() == 0.0
    assert env.current_player == 1
    assert env.winner == 0
    assert env.terminated is False


def test_legal_actions_tracks_empty_columns():
    """M020 / M055：legal_actions 返回的是“空列”，不是“有子的列”。"""
    env = ConnectFour()
    env.reset()
    assert env.legal_actions().tolist() == [0, 1, 2, 3, 4, 5, 6]

    env.board[0, 3] = 1
    assert env.legal_actions().tolist() == [0, 1, 2, 4, 5, 6]

    env.board[:, :] = 1
    assert env.legal_actions().size == 0


def test_step_returns_full_transition_tuple():
    """M044 / M034：合法落子必须返回 5 元组，且 invalid=False。"""
    env = ConnectFour()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(3)

    assert obs.shape == (2, 6, 7)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["invalid"] is False
    assert info["mover"] == 1
    assert info["winner"] is None
    # 换手后，观测应切到玩家 2 的视角
    assert obs[0].sum() == 0
    assert obs[1].sum() == 1


def test_step_illegal_move_penalizes_without_truncating():
    """M030 / M032：非法动作的 truncated、winner 字段与内部状态都要对。"""
    env = ConnectFour()
    env.reset()
    env.board[:, 0] = [1, 1, 1, 1, 1, 1]

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert reward < 0
    assert info["invalid"] is True
    assert info["winner"] == 2  # 当前行棋方是 1，判负给对手
    # 返回值说结束了，内部状态也必须一致，否则后续 step 会把这一局当成还在进行
    assert env.terminated is True
    assert env.winner == 2


def test_step_after_termination_keeps_winner():
    """已结束后再 step：不能把胜方改成对手。"""
    env = ConnectFour()
    env.reset()
    play(env, [0, 1, 0, 1, 0, 1, 0])
    assert env.terminated is True
    assert env.winner == 1

    _, reward, terminated, truncated, info = env.step(2)

    assert terminated is True
    assert truncated is False
    assert reward < 0
    assert info["invalid"] is True
    assert info["winner"] == 1
    assert env.winner == 1


@pytest.mark.parametrize("action", [7, -1])
def test_action_out_of_range_is_rejected(action):
    """M031：越界列号不能被 `_can_drop` 当成合法下标（会 IndexError）。"""
    env = ConnectFour()
    env.reset()

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert reward < 0
    assert info["invalid"] is True
    assert env.terminated is True


def test_three_in_a_row_does_not_end_the_game():
    """M081：这才是真正的漏洞——赢棋判定反转，旧测试也会“通过”。

    旧用例的循环一看到 terminated 就 break，所以“第一步就误判获胜”
    同样能让 winner == 1 成立。这里必须断言“三连时还没结束”。
    """
    env = ConnectFour()
    env.reset()
    play(env, [0, 1, 0, 1, 0, 1])

    assert env.terminated is False
    assert env.winner == 0

    env.step(0)  # 第四子落下才结束
    assert env.terminated is True
    assert env.winner == 1


def test_vertical_win_is_detected():
    """M006：竖方向 (1, 0) 被改坏后，三连会被当成结束或永远赢不了。"""
    env = ConnectFour()
    env.reset()
    play(env, [0, 1, 0, 1, 0, 1])
    assert env.terminated is False

    env.step(0)
    assert env.terminated is True
    assert env.winner == 1


def test_diagonal_down_right_win_is_detected():
    """M002 / M065：对角线方向缺失时，这里不会结束（或直接卡死）。"""
    env = ConnectFour()
    env.reset()
    play(env, [0, 3, 1, 3, 1, 3, 2, 6, 2, 6, 2, 6])
    assert env.terminated is False

    env.step(3)  # 玩家1 在 (2, 3) 完成 ↘ 四连
    assert env.terminated is True
    assert env.winner == 1


def test_diagonal_down_left_win_is_detected():
    """M004：↙ 方向同样要覆盖，否则 (1, -1) 写错也没人发现。"""
    env = ConnectFour()
    env.reset()
    play(env, [6, 3, 5, 3, 5, 3, 4, 0, 4, 0, 4, 0])
    assert env.terminated is False

    env.step(3)  # 玩家1 在 (2, 3) 完成 ↙ 四连
    assert env.terminated is True
    assert env.winner == 1


def test_column_with_five_stones_still_accepts_the_sixth():
    """M058：range 的终点写错时，只有“已经堆了 5 个”的列才会暴露。"""
    env = ConnectFour()
    env.reset()
    env.board[1:, 0] = [1, 2, 1, 2, 1]

    env.step(0)

    assert env.terminated is False
    assert env.board[0, 0] == 1
    assert env.board[1, 0] == 1
    assert env.board[5, 0] == 1


def test_full_board_without_four_is_a_draw():
    """M039 / M041：没有平局用例时，平局分支写错永远是绿的。"""
    env = ConnectFour()
    env.reset()
    env.board[:, :] = np.array(DRAW_BOARD, dtype=np.int8)
    env.board[0, 0] = 0  # 留最后一格
    env.current_player = 1

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert reward == 0.0
    assert env.winner == 0
    assert info["winner"] == 0


def test_top_row_win_is_detected():
    """M079：顶行（row 0）是边界，`0 <= r` 写成 `1 <= r` 时这里检不出。"""
    env = ConnectFour()
    env.reset()
    env.board[1:, :] = np.array(DRAW_BOARD[1:], dtype=np.int8)
    env.board[0, :] = [1, 1, 1, 0, 1, 2, 2]
    env.current_player = 1

    _, _, terminated, _, _ = env.step(3)

    assert terminated is True
    assert env.winner == 1


def test_render_uses_distinct_symbols():
    """M090：符号表键写错时，空格子会渲染成棋子。"""
    env = ConnectFour()
    env.reset()
    assert set(env.render().splitlines()) == {". . . . . . ."}

    env.step(3)  # 玩家1
    env.step(3)  # 玩家2
    text = env.render()
    assert "X" in text
    assert "O" in text
    assert env.is_single_agent is False
