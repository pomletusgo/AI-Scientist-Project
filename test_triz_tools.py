#!/usr/bin/env python3
"""
TRIZ Tools 测试用例 & 运行逻辑总结
===================================
覆盖 triz_tools.py 全部 6 个工具函数、3 个数据结构、工具注册机制。

用法:
    python test_triz_tools.py              # 运行所有测试
    python test_triz_tools.py --verbose    # 详细输出
    python test_triz_tools.py --logic      # 仅打印运行逻辑总结
"""

import json
import sys
import os

# Fix Windows GBK encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_scientist.triz_tools import (
    # === 数据结构 ===
    PARAMETERS,           # 39 工程参数
    ML_PARAMETER_MAP,     # ML 概念 → 参数 ID 映射
    PRINCIPLES,           # 40 发明原理
    MATRIX,               # 矛盾矩阵
    # === 工具函数 ===
    lookup_contradiction_matrix,
    get_principle_detail,
    map_ml_concept_to_parameters,
    evaluate_ideality,
    check_principle_novelty,
    suggest_contradictions,
    # === 工具注册 ===
    TOOLS_DEFINITION,
    TOOL_MAP,
)


# ============================================================
# 运行逻辑总结
# ============================================================

LOGIC_SUMMARY = r"""
╔══════════════════════════════════════════════════════════════════╗
║           TRIZ Tools 内部运行逻辑总结                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │              三 层 数 据 结 构                            │    ║
║  ├─────────────────────────────────────────────────────────┤    ║
║  │ Layer 1: PARAMETERS (39 工程参数)                         │    ║
║  │   例: #28="Measurement accuracy", #35="Adaptability"      │    ║
║  │   用途: 定义矛盾双方的语言                                  │    ║
║  │                                                          │    ║
║  │ Layer 2: MATRIX (矛盾矩阵)                                │    ║
║  │   键: (改善参数, 恶化参数) → 值: [推荐原理编号]             │    ║
║  │   例: (28, 25) → [24, 34, 28, 32]                        │    ║
║  │   含义: 改善精度(28)却损失时间(25)时, 用原理 24,34,28,32   │    ║
║  │   预设了 23 组 ML 常见矛盾                                  │    ║
║  │                                                          │    ║
║  │ Layer 3: PRINCIPLES (40 发明原理)                         │    ║
║  │   每条包含: (名称, 描述, ML应用)                            │    ║
║  │   例: #35=("Parameter Change", "...",                     │    ║
║  │           "Temperature scaling, LR schedules...")         │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │              六 个 工 具 函 数                             │    ║
║  ├─────────────────────────────────────────────────────────┤    ║
║  │                                                          │    ║
║  │ ① suggest_contradictions(domain, task_description)       │    ║
║  │    输入: ML 领域名 (如 "classification", "nlp")            │    ║
║  │    流程: 匹配预设 → 对每个矛盾调用 lookup_contradiction_   │    ║
║  │           matrix 获取推荐原理                               │    ║
║  │    输出: [{improving, worsening, statement,               │    ║
║  │            recommended_principles}]                        │    ║
║  │                                                          │    ║
║  │ ② lookup_contradiction_matrix(improving, worsening)      │    ║
║  │    输入: 两个参数 ID (1-39)                                │    ║
║  │    流程: MATRIX.get((improving, worsening))               │    ║
║  │          若未命中 → 返回默认值 [1, 35, 15]                  │    ║
║  │    输出: {contradiction: {...}, recommended_principles}   │    ║
║  │                                                          │    ║
║  │ ③ get_principle_detail(principle_number)                 │    ║
║  │    输入: 原理编号 (1-40)                                   │    ║
║  │    流程: PRINCIPLES.get(n) → 拼接 how_to_apply 指导语      │    ║
║  │    输出: {number, name, description, ml_applications,     │    ║
║  │            how_to_apply}                                   │    ║
║  │                                                          │    ║
║  │ ④ check_principle_novelty(principle_number, domain)      │    ║
║  │    输入: 原理编号 + 领域名                                  │    ║
║  │    流程: 判断是否在 COMMON_ML_PRINCIPLES 集合中             │    ║
║  │          COMMON: {1,2,5,7,10,13,15,23,25,26,28,35,40}    │    ║
║  │          RARE:  其他 27 个                                 │    ║
║  │    输出: {principle, domain, novelty_assessment}          │    ║
║  │                                                          │    ║
║  │ ⑤ evaluate_ideality(benefits, costs, harms)              │    ║
║  │    输入: 三组 {name, score} 列表 (score: 1-10)             │    ║
║  │    流程: sum(benefits) / (sum(costs) + sum(harms))        │    ║
║  │          分母为 0 时兜底为 1                                │    ║
║  │    输出: {benefits_total, costs_total, harms_total,       │    ║
║  │            ideality_score, level, formula}                 │    ║
║  │    评级: ≥2.0=Excellent | ≥1.0=Good | ≥0.5=Marginal       │    ║
║  │                                                          │    ║
║  │ ⑥ map_ml_concept_to_parameters(concept)                  │    ║
║  │    输入: ML 概念词 (如 "accuracy", "overfitting")          │    ║
║  │    流程: 在 ML_PARAMETER_MAP 中模糊匹配                    │    ║
║  │    输出: {concept, matches: [{keyword, parameter_id,      │    ║
║  │            parameter_name}]} 或带 suggestion 的错误信息     │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │              工 具 注 册 机 制                             │    ║
║  ├─────────────────────────────────────────────────────────┤    ║
║  │ TOOLS_DEFINITION: OpenAI function-calling 格式             │    ║
║  │   6 个 JSON Schema 定义 (name, description, parameters)    │    ║
║  │   → 传给 LLM 的 tools 参数                                  │    ║
║  │                                                          │    ║
║  │ TOOL_MAP: 函数名 → 可调用对象                               │    ║
║  │   → LLM 返回 tool_call 后, 路由到对应函数执行               │    ║
║  │                                                          │    ║
║  │ 调用链:                                                    │    ║
║  │   LLM 输出 tool_call                                       │    ║
║  │   → get_response_with_tools() 解析                         │    ║
║  │   → TOOL_MAP[func_name](**args)                           │    ║
║  │   → 结果注入 msg_history                                   │    ║
║  │   → LLM 继续决策 (调用更多工具 or 输出最终答案)              │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │              典 型 调 用 流 程                             │    ║
║  ├─────────────────────────────────────────────────────────┤    ║
║  │                                                          │    ║
║  │  LLM 收到研究任务                                          │    ║
║  │    ↓                                                     │    ║
║  │  ① suggest_contradictions("classification")               │    ║
║  │     → 3 个矛盾, 每个带推荐原理                              │    ║
║  │    ↓                                                     │    ║
║  │  ② 对每个矛盾: lookup_contradiction_matrix(39, 27)        │    ║
║  │     → 精确查询改善某参数会恶化哪个参数                       │    ║
║  │    ↓                                                     │    ║
║  │  ③ 对推荐原理: get_principle_detail(35)                    │    ║
║  │     → 完整描述 + ML 应用场景 + 操作指南                     │    ║
║  │    ↓                                                     │    ║
║  │  ④ check_principle_novelty(35, "classification")          │    ║
║  │     → 新颖性评估 (高→论文贡献大, 中→需组合)                │    ║
║  │    ↓                                                     │    ║
║  │  ⑤ evaluate_ideality(benefits, costs, harms)              │    ║
║  │     → 量化方案优劣 (Ideality = 收益/(成本+副作用))         │    ║
║  │    ↓                                                     │    ║
║  │  ⑥ map_ml_concept_to_parameters("accuracy")               │    ║
║  │     → 辅助工具: ML 术语 ↔ 工程参数 ID 互转                 │    ║
║  └─────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ============================================================
# 测试用例
# ============================================================

def test_parameters_data():
    """测试 PARAMETERS 数据结构完整性"""
    assert len(PARAMETERS) == 39, f"Expected 39 parameters, got {len(PARAMETERS)}"
    assert 1 in PARAMETERS and 39 in PARAMETERS, "Missing boundary parameters"
    assert PARAMETERS[28] == "Measurement accuracy"
    assert PARAMETERS[39] == "Productivity"
    assert PARAMETERS[9] == "Speed"
    assert PARAMETERS[25] == "Loss of Time"
    print("  [PASS] PARAMETERS: 39 entries, correct values at key positions")


def test_principles_data():
    """测试 PRINCIPLES 数据结构完整性"""
    assert len(PRINCIPLES) == 40, f"Expected 40 principles, got {len(PRINCIPLES)}"
    # 每条是 (name, description, ml_applications) 三元组
    for n, entry in PRINCIPLES.items():
        assert isinstance(entry, tuple), f"Principle #{n} should be tuple"
        assert len(entry) == 3, f"Principle #{n} should have 3 elements (name, desc, ml_app)"
        assert entry[0], f"Principle #{n} has empty name"
        assert entry[1], f"Principle #{n} has empty description"
        assert entry[2], f"Principle #{n} has empty ML applications"
    # Spot checks
    assert PRINCIPLES[1][0] == "Segmentation"
    assert PRINCIPLES[40][0] == "Composite Materials"
    assert "Mixture-of-Experts" in PRINCIPLES[1][2]
    print("  [PASS] PRINCIPLES: 40 entries, all tuples with 3 elements, key values correct")


def test_matrix_data():
    """测试 MATRIX 矛盾矩阵"""
    assert len(MATRIX) >= 20, f"Expected >=20 matrix entries, got {len(MATRIX)}"
    # 验证键格式
    for key, value in MATRIX.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert 1 <= key[0] <= 39, f"Bad improving param: {key[0]}"
        assert 1 <= key[1] <= 39, f"Bad worsening param: {key[1]}"
        assert isinstance(value, list) and len(value) >= 2
        for p in value:
            assert 1 <= p <= 40, f"Bad principle number: {p}"
    # 关键矛盾验证
    assert (9, 25) in MATRIX, "Missing Speed vs Loss of Time"
    assert (39, 27) in MATRIX, "Missing Productivity vs Reliability"
    print(f"  [PASS] MATRIX: {len(MATRIX)} entries, all valid (param, param) → [principles]")


def test_ml_parameter_map():
    """测试 ML_PARAMETER_MAP"""
    assert len(ML_PARAMETER_MAP) >= 15
    assert ML_PARAMETER_MAP["accuracy"] == 28
    assert ML_PARAMETER_MAP["speed"] == 9
    assert ML_PARAMETER_MAP["robustness"] == 27
    assert ML_PARAMETER_MAP["complexity"] == 36
    assert ML_PARAMETER_MAP["generalization"] == 35
    print(f"  [PASS] ML_PARAMETER_MAP: {len(ML_PARAMETER_MAP)} entries, key mappings correct")


# ----- 工具函数测试 -----

def test_lookup_contradiction_matrix():
    """测试 lookup_contradiction_matrix"""
    # 命中
    result = lookup_contradiction_matrix(28, 25)  # Accuracy vs Loss of Time
    assert result["contradiction"]["improving"]["name"] == "Measurement accuracy"
    assert result["contradiction"]["worsening"]["name"] == "Loss of Time"
    assert len(result["recommended_principles"]) >= 2
    print(f"    命中: 改善{result['contradiction']['improving']['name']} "
          f"→ 恶化{result['contradiction']['worsening']['name']} "
          f"→ 原理 {[p['number'] for p in result['recommended_principles']]}")

    # 未命中 (fallback)
    result2 = lookup_contradiction_matrix(1, 2)  # 矩阵中没有的直接组合
    assert result2["recommended_principles"][0]["number"] in (1, 35, 15)
    print(f"    未命中(兜底): 改善#{1} 恶化#{2} → 默认原理 {[p['number'] for p in result2['recommended_principles']]}")

    # Productivity vs Reliability
    result3 = lookup_contradiction_matrix(39, 27)
    assert result3["contradiction"]["improving"]["name"] == "Productivity"
    print(f"    高价值组合: 改善{result3['contradiction']['improving']['name']} "
          f"恶化{result3['contradiction']['worsening']['name']} "
          f"→ {[p['number'] for p in result3['recommended_principles']]}")
    print("  [PASS] lookup_contradiction_matrix: hit, miss-fallback, key pairs all correct")


def test_get_principle_detail():
    """测试 get_principle_detail"""
    # 正常
    result = get_principle_detail(35)
    assert result["number"] == 35
    assert result["name"] == "Parameter Change"
    assert "how_to_apply" in result
    assert "Temperature scaling" in result["ml_applications"]
    print(f"    原理#35: {result['name']} → {result['ml_applications'][:60]}...")

    # 错误编号
    result2 = get_principle_detail(999)
    assert "error" in result2
    print(f"    错误#999: {result2['error']}")

    # 边界
    result3 = get_principle_detail(1)
    assert result3["name"] == "Segmentation"
    result4 = get_principle_detail(40)
    assert result4["name"] == "Composite Materials"
    print(f"    边界: #1={result3['name']}, #40={result4['name']}")
    print("  [PASS] get_principle_detail: valid, invalid, boundaries all correct")


def test_map_ml_concept_to_parameters():
    """测试 map_ml_concept_to_parameters"""
    # 精确命中
    result = map_ml_concept_to_parameters("accuracy")
    assert result["concept"] == "accuracy"
    assert len(result["matches"]) >= 1
    assert any(m["parameter_id"] == 28 for m in result["matches"])
    print(f"    'accuracy' → parameter_ids={[m['parameter_id'] for m in result['matches']]}")

    # 模糊匹配 (关键词包含)
    result2 = map_ml_concept_to_parameters("model overfitting problem")
    assert any(m["parameter_id"] == 13 for m in result2.get("matches", []))
    print(f"    'model overfitting problem' → {len(result2.get('matches',[]))} matches")

    # 无匹配
    result3 = map_ml_concept_to_parameters("quantum entanglement")
    assert "suggestion" in result3
    print(f"    'quantum entanglement' → suggestion returned")

    # 常见概念
    for concept, expected_id in [("latency", 25), ("model_size", 26), ("fairness", 30)]:
        r = map_ml_concept_to_parameters(concept)
        assert any(m["parameter_id"] == expected_id for m in r.get("matches", []))
    print("  [PASS] map_ml_concept_to_parameters: exact, fuzzy, no-match, key concepts all correct")


def test_evaluate_ideality():
    """测试 evaluate_ideality"""
    # 理想方案: 高收益, 低成本, 无副作用
    result = evaluate_ideality(
        benefits=[{"name": "accuracy gain", "score": 9},
                   {"name": "robustness", "score": 8}],
        costs=[{"name": "compute overhead", "score": 2}],
        harms=[{"name": "overfitting risk", "score": 1}],
    )
    # Ideality = (9+8) / (2+1) = 17/3 ≈ 5.667
    assert abs(result["ideality_score"] - 5.667) < 0.01
    assert result["level"] == "Excellent — strongly recommend pursuing"
    print(f"    理想方案: {result['formula']} → {result['level']}")

    # 一般方案
    result2 = evaluate_ideality(
        benefits=[{"name": "small improvement", "score": 5}],
        costs=[{"name": "complexity", "score": 6}],
        harms=[{"name": "maintenance burden", "score": 4}],
    )
    # Ideality = 5 / (6+4) = 0.5
    assert abs(result2["ideality_score"] - 0.5) < 0.01
    assert result2["level"] == "Marginal — needs refinement to reduce costs/harm"
    print(f"    一般方案: {result2['formula']} → {result2['level']}")

    # 零成本场景 (分母兜底)
    result3 = evaluate_ideality(
        benefits=[{"name": "free improvement", "score": 8}],
        costs=[],
        harms=[],
    )
    # Ideality = 8 / 1 = 8.0 (denominator floor at 1)
    assert result3["ideality_score"] == 8.0
    print(f"    零成本: {result3['formula']} → {result3['level']}")

    # 负方案
    result4 = evaluate_ideality(
        benefits=[{"name": "minor gain", "score": 3}],
        costs=[{"name": "huge complexity", "score": 10}],
        harms=[{"name": "security risk", "score": 9}],
    )
    # Ideality = 3 / (10+9) = 3/19 ≈ 0.158
    assert result4["ideality_score"] < 0.5
    assert "Poor" in result4["level"]
    print(f"    负方案: {result4['formula']} → {result4['level']}")
    print("  [PASS] evaluate_ideality: excellent, marginal, zero-cost, poor — all correct")


def test_check_principle_novelty():
    """测试 check_principle_novelty"""
    # 常见原理 (COMMON_ML_PRINCIPLES)
    result = check_principle_novelty(1, "classification")  # Segmentation
    assert "HIGH" not in result["novelty_assessment"]  # common → moderate
    print(f"    #1 Segregation in classification: {result['novelty_assessment'][:50]}...")

    # 罕见原理 (RARE_ML_PRINCIPLES)
    result2 = check_principle_novelty(14, "nlp")  # Spheroidality
    assert "HIGH" in result2["novelty_assessment"]
    print(f"    #14 Spheroidality in NLP: {result2['novelty_assessment'][:50]}...")

    # 常见原理
    result3 = check_principle_novelty(35, "vision")  # Parameter Change (common)
    assert "HIGH" not in result3["novelty_assessment"]
    print(f"    #35 Parameter Change in vision: {result3['novelty_assessment'][:50]}...")

    # 罕见原理
    result4 = check_principle_novelty(39, "regression")  # Inert Atmosphere (rare)
    assert "HIGH" in result4["novelty_assessment"]
    print(f"    #39 Inert Atmosphere in regression: {result4['novelty_assessment'][:50]}...")

    # 统计: 通过遍历验证 40 个原理都能被分类
    common_count = 0
    rare_count = 0
    for n in range(1, 41):
        r = check_principle_novelty(n, "test")
        if "HIGH" in r["novelty_assessment"]:
            rare_count += 1
        else:
            common_count += 1
    assert common_count + rare_count == 40
    print(f"    分类覆盖: {common_count} common + {rare_count} rare = 40 total [PASS]")
    print("  [PASS] check_principle_novelty: common, rare, complete coverage")


def test_suggest_contradictions():
    """测试 suggest_contradictions"""
    # 已知领域
    result = suggest_contradictions("classification")
    assert "contradictions" in result
    assert len(result["contradictions"]) == 3
    for c in result["contradictions"]:
        assert "improving" in c and "worsening" in c and "statement" in c
        assert "recommended_principles" in c
        assert len(c["recommended_principles"]) >= 2
    print(f"    'classification' → {len(result['contradictions'])} contradictions:")
    for c in result["contradictions"]:
        print(f"      {c['statement'][:80]} → principles={[p['number'] for p in c['recommended_principles']]}")

    # 模糊匹配
    result2 = suggest_contradictions("binary classification task")
    assert result2["domain"] == "binary classification task"
    assert len(result2["contradictions"]) == 3
    print(f"    'binary classification task' → matched classification preset")

    # 未知领域 (fallback to classification)
    result3 = suggest_contradictions("quantum computing")
    assert len(result3["contradictions"]) == 3
    print(f"    'quantum computing' → fallback to classification preset")

    # NLP
    result4 = suggest_contradictions("nlp")
    assert any("language" in c["statement"].lower() for c in result4["contradictions"])
    print(f"    'nlp' → domain-specific contradictions present")

    # 所有预设领域
    for domain in ["churn", "classification", "detection", "regression", "nlp", "vision"]:
        r = suggest_contradictions(domain)
        assert len(r["contradictions"]) == 3
    print(f"    All 6 presets produce exactly 3 contradictions each [PASS]")
    print("  [PASS] suggest_contradictions: all presets, fuzzy match, fallback, structure correct")


def test_tool_registration():
    """测试工具注册机制"""
    # TOOLS_DEFINITION
    assert len(TOOLS_DEFINITION) == 6
    tool_names = [t["function"]["name"] for t in TOOLS_DEFINITION]
    expected_names = [
        "suggest_contradictions",
        "lookup_contradiction_matrix",
        "get_principle_detail",
        "check_principle_novelty",
        "evaluate_ideality",
        "map_ml_concept_to_parameters",
    ]
    assert tool_names == expected_names, f"Tool names mismatch: {tool_names}"

    # 每个定义都有必需的字段
    for tool in TOOLS_DEFINITION:
        assert tool["type"] == "function"
        func = tool["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        if "required" in params:  # Optional field
            for req in params["required"]:
                assert req in params["properties"]
    print(f"  [PASS] TOOLS_DEFINITION: {len(TOOLS_DEFINITION)} tools, valid OpenAI function-calling schema")

    # TOOL_MAP
    assert len(TOOL_MAP) == 6
    for name in expected_names:
        assert name in TOOL_MAP, f"Missing {name} in TOOL_MAP"
        assert callable(TOOL_MAP[name]), f"{name} is not callable"
    print(f"  [PASS] TOOL_MAP: {len(TOOL_MAP)} entries, all callable")

    # 验证 TOOLS_DEFINITION 和 TOOL_MAP 一致性
    map_names = set(TOOL_MAP.keys())
    defn_names = set(t["function"]["name"] for t in TOOLS_DEFINITION)
    assert map_names == defn_names, f"Mismatch: TOOL_MAP has {map_names - defn_names}, TOOLS_DEFINITION has {defn_names - map_names}"
    print(f"  [PASS] TOOLS_DEFINITION ↔ TOOL_MAP: names perfectly aligned")


def test_end_to_end_flow():
    """端到端调用流程模拟"""
    print("  Simulating full LLM tool-call flow...")

    # Step 1: LLM 拿到研究任务, 调用 suggest_contradictions
    contradictions = suggest_contradictions("classification", "Improve model accuracy without sacrificing speed")
    assert len(contradictions["contradictions"]) == 3

    # Step 2: 对第一个矛盾, 查询矩阵
    c1 = contradictions["contradictions"][0]
    matrix_result = lookup_contradiction_matrix(c1["improving"], c1["worsening"])
    assert len(matrix_result["recommended_principles"]) >= 2

    # Step 3: 对推荐的第一个原理, 查看详情
    p1 = matrix_result["recommended_principles"][0]["number"]
    detail = get_principle_detail(p1)
    assert detail["number"] == p1
    assert "how_to_apply" in detail

    # Step 4: 检查新颖性
    novelty = check_principle_novelty(p1, "classification")
    assert "novelty_assessment" in novelty

    # Step 5: 评分
    ideality = evaluate_ideality(
        benefits=[{"name": "higher accuracy", "score": 8}],
        costs=[{"name": "training time", "score": 4}],
        harms=[{"name": "slight overfitting", "score": 2}],
    )
    # 8 / (4+2) = 8/6 ≈ 1.333 → Good
    assert ideality["ideality_score"] >= 1.0

    # Step 6: 辅助映射
    mapping = map_ml_concept_to_parameters("accuracy")
    assert len(mapping["matches"]) >= 1

    print(f"  [PASS] End-to-end flow: 6 steps, all successful")
    print(f"    Contradiction: {c1['statement'][:80]}")
    print(f"    Principle #{p1}: {detail['name']}")
    print(f"    Novelty: {novelty['novelty_assessment'][:60]}")
    print(f"    Ideality: {ideality['formula']} = {ideality['level']}")


# ============================================================
# 测试运行器
# ============================================================

def run_all_tests(verbose=False):
    """运行所有测试用例"""
    tests = [
        # 数据结构
        ("PARAMETERS 数据结构", test_parameters_data),
        ("PRINCIPLES 数据结构", test_principles_data),
        ("MATRIX 矛盾矩阵", test_matrix_data),
        ("ML_PARAMETER_MAP 映射表", test_ml_parameter_map),
        # 工具函数
        ("lookup_contradiction_matrix", test_lookup_contradiction_matrix),
        ("get_principle_detail", test_get_principle_detail),
        ("map_ml_concept_to_parameters", test_map_ml_concept_to_parameters),
        ("evaluate_ideality", test_evaluate_ideality),
        ("check_principle_novelty", test_check_principle_novelty),
        ("suggest_contradictions", test_suggest_contradictions),
        # 工具注册
        ("TOOLS_DEFINITION + TOOL_MAP", test_tool_registration),
        # 端到端
        ("End-to-End Flow", test_end_to_end_flow),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if verbose:
                print(f"\n[{name}]")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  [FAIL] [{name}] FAILED: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TRIZ Tools test suite & logic summary")
    parser.add_argument("--verbose", action="store_true", help="Detailed output for each test")
    parser.add_argument("--logic", action="store_true", help="Print logic summary only, skip tests")
    args = parser.parse_args()

    if args.logic:
        print(LOGIC_SUMMARY)
    else:
        print("=" * 60)
        print("TRIZ Tools — Test Suite")
        print("=" * 60)
        success = run_all_tests(verbose=args.verbose)
        print()
        print("Tip: use --logic to see the full logic summary")
        print("     use --verbose for detailed test output")
        sys.exit(0 if success else 1)
