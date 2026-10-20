"""Prior lessons management: search, conflict detection, and lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from magnetic_vasp_agent.input_review_models import InputReviewRecord, PriorLesson


class LessonConflict:
    """表示两条相互冲突的经验"""

    def __init__(self, lessons: list[PriorLesson], common_tags: set[str], reason: str):
        self.lessons = lessons
        self.common_tags = common_tags
        self.reason = reason

    @property
    def lesson1(self) -> PriorLesson:
        """向后兼容属性"""
        return self.lessons[0]

    @property
    def lesson2(self) -> PriorLesson:
        """向后兼容属性"""
        return self.lessons[1]

    def __str__(self) -> str:
        return (
            f"冲突检测 [{self.reason}]:\n"
            f"  经验1 ({self.lesson1.workflow_id}): {self.lesson1.lesson}\n"
            f"  经验2 ({self.lesson2.workflow_id}): {self.lesson2.lesson}"
        )


class LessonManager:
    """管理审核经验的生命周期和查询"""

    def __init__(self, reviews_path: Path):
        self.reviews_path = reviews_path

    def load_all_lessons(
        self, include_deprecated: bool = False, max_age_months: int | None = 6
    ) -> list[PriorLesson]:
        """加载所有审核经验

        Args:
            include_deprecated: 是否包含已过期的经验
            max_age_months: 最大年龄（月），None 表示不限制
        """
        lessons = []
        if not self.reviews_path.exists():
            return lessons

        with self.reviews_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = InputReviewRecord.model_validate_json(line)
                    if record.decision and record.lesson:
                        lesson = PriorLesson(
                            workflow_id=record.workflow_id,
                            decision=record.decision,
                            reviewer=record.reviewer,
                            lesson=record.lesson,
                            tags=record.tags,
                            timestamp=record.recorded_at,
                            deprecated=record.deprecated,
                            deprecation_reason=record.deprecation_reason,
                        )

                        # 过滤条件
                        if not include_deprecated and lesson.deprecated:
                            continue
                        if (
                            max_age_months is not None
                            and not lesson.deprecated
                            and not lesson.is_recent(max_age_months * 30)
                        ):
                            continue

                        lessons.append(lesson)
                except Exception:
                    continue

        return lessons

    def load_recent_lessons(self, max_age_days: int) -> list[PriorLesson]:
        """加载指定天数内的经验
        
        Args:
            max_age_days: 最大年龄（天）
        """
        all_lessons = self.load_all_lessons(include_deprecated=False, max_age_months=None)
        return [lesson for lesson in all_lessons if lesson.is_recent(max_age_days)]

    def search_lessons(
        self,
        species: list[str] | None = None,
        tags: list[str] | None = None,
        decision: Literal["accept", "reject", "revise"] | None = None,
        reviewer: str | None = None,
        text_query: str | None = None,
        max_age_days: int | None = None,
        limit: int = 20,
        include_deprecated: bool = False,
    ) -> list[PriorLesson]:
        """搜索符合条件的经验

        Args:
            species: 物种列表（例如 ["Fe", "O"]）
            tags: 标签过滤
            decision: 决策类型过滤
            reviewer: 审核员过滤
            text_query: 文本关键词搜索（不区分大小写）
            max_age_days: 最大年龄（天）
            limit: 返回结果数量限制
            include_deprecated: 是否包含已废弃的经验
        """
        # 映射决策值
        decision_map = {"accepted": "accept", "rejected": "reject"}
        if decision:
            decision = decision_map.get(decision, decision)
        
        lessons = self.load_all_lessons(include_deprecated=include_deprecated)
        results = []

        for lesson in lessons:
            # 决策过滤
            if decision and lesson.decision != decision:
                continue

            # 审核员过滤
            if reviewer and lesson.reviewer != reviewer:
                continue

            # 标签过滤
            if tags and not any(tag in lesson.tags for tag in tags):
                continue

            # 文本搜索
            if text_query:
                query_lower = text_query.lower()
                if query_lower not in lesson.lesson.lower():
                    continue

            # 年龄过滤
            if max_age_days is not None and not lesson.is_recent(max_age_days):
                continue

            # 物种过滤（需要从原始记录中提取）
            if species:
                if not self._lesson_matches_species(lesson, species):
                    continue

            results.append(lesson)

            if len(results) >= limit:
                break

        return results

    def _lesson_matches_species(
        self, lesson: PriorLesson, species: list[str]
    ) -> bool:
        """检查经验是否匹配指定物种"""
        # 从原始记录中查找物种信息
        if not self.reviews_path.exists():
            return False

        with self.reviews_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = InputReviewRecord.model_validate_json(line)
                    if record.workflow_id == lesson.workflow_id:
                        # 检查所有步骤的物种
                        for step in record.steps:
                            step_species = set(step.species_order)
                            query_species = set(species)
                            if step_species == query_species:
                                return True
                        return False
                except Exception:
                    continue
        return False

    def detect_conflicts(
        self, lessons: list[PriorLesson] | None = None
    ) -> list[LessonConflict]:
        """检测经验中的冲突

        冲突定义：
        1. 相同标签但决策相反（accept vs reject/revise）
        2. 文本内容明显矛盾（例如："必须使用 U" vs "不要使用 U"）
        """
        if lessons is None:
            lessons = self.load_all_lessons()

        conflicts = []

        for i, l1 in enumerate(lessons):
            for l2 in lessons[i + 1 :]:
                # 检查标签重叠且决策相反
                if set(l1.tags) & set(l2.tags):
                    if (l1.decision == "accept" and l2.decision in ["reject", "revise"]) or (
                        l2.decision == "accept" and l1.decision in ["reject", "revise"]
                    ):
                        common_tags = set(l1.tags) & set(l2.tags)
                        conflicts.append(
                            LessonConflict(
                                lessons=[l1, l2],
                                common_tags=common_tags,
                                reason=f"相同标签 {common_tags} 但决策相反",
                            )
                        )
                    # 检查相同标签且相同决策的竞争建议
                    elif set(l1.tags) == set(l2.tags) and l1.decision == l2.decision == "accept":
                        common_tags = set(l1.tags)
                        conflicts.append(
                            LessonConflict(
                                lessons=[l1, l2],
                                common_tags=common_tags,
                                reason=f"相同标签 {common_tags} 的竞争建议",
                            )
                        )

                # 检查文本矛盾
                if self._text_contradicts(l1.lesson, l2.lesson):
                    conflicts.append(
                        LessonConflict(
                            lessons=[l1, l2],
                            common_tags=set(l1.tags) & set(l2.tags),
                            reason="文本内容可能存在矛盾",
                        )
                    )

        return conflicts

    def _text_contradicts(self, text1: str, text2: str) -> bool:
        """简单的文本矛盾检测"""
        # 简化版：检测否定词对
        positive_negative_pairs = [
            ("必须", "不要"),
            ("需要", "禁止"),
            ("应该", "不应"),
            ("建议", "避免"),
        ]

        for pos, neg in positive_negative_pairs:
            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                return True

        return False

    def mark_deprecated(
        self, workflow_id: str, reason: str, output_path: Path | None = None
    ) -> bool:
        """标记经验为过期状态

        直接更新原始文件中的记录
        """
        if not self.reviews_path.exists():
            return False

        # 读取所有记录
        all_records = []
        found = False
        with self.reviews_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = InputReviewRecord.model_validate_json(line)
                    if record.workflow_id == workflow_id:
                        record.deprecated = True
                        record.deprecation_reason = reason
                        found = True
                    all_records.append(record)
                except Exception:
                    continue

        if not found:
            return False

        # 写回所有记录
        with self.reviews_path.open("w", encoding="utf-8") as f:
            for record in all_records:
                f.write(record.model_dump_json() + "\n")

        return True

    def get_review_metrics(self) -> dict:
        """生成审核统计指标"""
        lessons = self.load_all_lessons(include_deprecated=True, max_age_months=None)

        if not lessons:
            return {
                "total_reviews": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "by_reviewer": {},
                "by_tag": {},
                "acceptance_rate": 0.0,
            }

        # 按决策统计
        by_decision = {}
        for lesson in lessons:
            by_decision[lesson.decision] = by_decision.get(lesson.decision, 0) + 1

        # 按审核员统计
        by_reviewer = {}
        for lesson in lessons:
            reviewer = lesson.reviewer or "unknown"
            by_reviewer[reviewer] = by_reviewer.get(reviewer, 0) + 1

        # 按标签统计
        by_tag = {}
        for lesson in lessons:
            for tag in lesson.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        # 接受率
        accept_count = by_decision.get("accept", 0)
        acceptance_rate = accept_count / len(lessons) if lessons else 0.0

        return {
            "total_reviews": len(lessons),
            "accepted_count": by_decision.get("accept", 0),
            "rejected_count": by_decision.get("reject", 0),
            "by_reviewer": {r: {"count": c} for r, c in by_reviewer.items()},
            "by_tag": by_tag,
            "acceptance_rate": acceptance_rate,
        }
