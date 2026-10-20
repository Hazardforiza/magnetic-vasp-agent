"""CLI commands for managing prior lessons."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from magnetic_vasp_agent.lesson_manager import LessonManager

lesson_app = typer.Typer(
    no_args_is_help=True, help="Search and manage prior input review lessons."
)
console = Console()


@lesson_app.command("search")
def search_lessons(
    reviews_file: Path = typer.Option(
        Path("var/reviews/input-reviews.jsonl"),
        "--reviews",
        help="Path to input reviews JSONL file",
    ),
    species: str = typer.Option(None, "--species", help="Filter by species (e.g., 'Fe,O')"),
    tags: str = typer.Option(None, "--tags", help="Filter by tags (comma-separated)"),
    decision: str = typer.Option(
        None, "--decision", help="Filter by decision (accept/reject/revise)"
    ),
    reviewer: str = typer.Option(None, "--reviewer", help="Filter by reviewer name"),
    query: str = typer.Option(None, "--query", "-q", help="Text search in lesson content"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum results to show"),
) -> None:
    """Search prior lessons with filters.

    Examples:
        magvasp lesson search --species Fe,O --tags hubbard
        magvasp lesson search --query "U_eff" --decision accept
        magvasp lesson search --reviewer alice --limit 10
    """
    manager = LessonManager(reviews_file)

    species_list = species.split(",") if species else None
    tags_list = tags.split(",") if tags else None

    results = manager.search_lessons(
        species=species_list,
        tags=tags_list,
        decision=decision,  # type: ignore
        reviewer=reviewer,
        text_query=query,
        limit=limit,
    )

    if not results:
        console.print("[yellow]未找到匹配的经验记录[/yellow]")
        return

    console.print(f"\n找到 {len(results)} 条经验记录:\n")

    for i, lesson in enumerate(results, 1):
        console.print(f"[bold cyan]{i}. {lesson.workflow_id}[/bold cyan]")
        console.print(f"   决策: [green]{lesson.decision}[/green]")
        if lesson.reviewer:
            console.print(f"   审核员: {lesson.reviewer}")
        if lesson.tags:
            console.print(f"   标签: {', '.join(lesson.tags)}")
        if lesson.timestamp:
            console.print(f"   时间: {lesson.timestamp}")
        console.print(f"   经验: {lesson.lesson}")
        if lesson.deprecated:
            console.print(f"   [red]已过期: {lesson.deprecation_reason}[/red]")
        console.print()


@lesson_app.command("conflicts")
def detect_conflicts(
    reviews_file: Path = typer.Option(
        Path("var/reviews/input-reviews.jsonl"),
        "--reviews",
        help="Path to input reviews JSONL file",
    ),
) -> None:
    """Detect conflicting lessons in the review history.

    Conflicts include:
    - Same tags but opposite decisions (accept vs reject/revise)
    - Contradictory text content
    """
    manager = LessonManager(reviews_file)
    conflicts = manager.detect_conflicts()

    if not conflicts:
        console.print("[green]✓ 未检测到经验冲突[/green]")
        return

    console.print(f"\n[yellow]检测到 {len(conflicts)} 个潜在冲突:[/yellow]\n")

    for i, conflict in enumerate(conflicts, 1):
        console.print(f"[bold red]冲突 {i}: {conflict.reason}[/bold red]")
        console.print(f"  经验1 ({conflict.lesson1.workflow_id}):")
        console.print(f"    决策: {conflict.lesson1.decision}")
        console.print(f"    内容: {conflict.lesson1.lesson}")
        console.print(f"  经验2 ({conflict.lesson2.workflow_id}):")
        console.print(f"    决策: {conflict.lesson2.decision}")
        console.print(f"    内容: {conflict.lesson2.lesson}")
        console.print()


@lesson_app.command("metrics")
def show_metrics(
    reviews_file: Path = typer.Option(
        Path("var/reviews/input-reviews.jsonl"),
        "--reviews",
        help="Path to input reviews JSONL file",
    ),
) -> None:
    """Show review statistics and metrics.

    Displays:
    - Total reviews
    - Distribution by decision (accept/reject/revise)
    - Distribution by reviewer
    - Top tags
    - Overall acceptance rate
    """
    manager = LessonManager(reviews_file)
    metrics = manager.get_review_metrics()

    if metrics["total"] == 0:
        console.print("[yellow]暂无审核记录[/yellow]")
        return

    console.print("\n[bold]审核统计总览[/bold]\n")
    console.print(f"总计: {metrics['total']} 条记录")
    console.print(f"接受率: {metrics['acceptance_rate']:.1%}\n")

    # 决策分布
    if metrics["by_decision"]:
        table_decision = Table(title="决策分布")
        table_decision.add_column("决策", style="cyan")
        table_decision.add_column("数量", justify="right", style="green")
        for decision, count in sorted(
            metrics["by_decision"].items(), key=lambda x: -x[1]
        ):
            table_decision.add_row(decision, str(count))
        console.print(table_decision)
        console.print()

    # 审核员分布
    if metrics["by_reviewer"]:
        table_reviewer = Table(title="审核员分布")
        table_reviewer.add_column("审核员", style="cyan")
        table_reviewer.add_column("审核数", justify="right", style="green")
        for reviewer, count in sorted(
            metrics["by_reviewer"].items(), key=lambda x: -x[1]
        )[:10]:  # 显示前10位
            table_reviewer.add_row(reviewer, str(count))
        console.print(table_reviewer)
        console.print()

    # 标签分布
    if metrics["by_tag"]:
        table_tag = Table(title="常用标签 (Top 10)")
        table_tag.add_column("标签", style="cyan")
        table_tag.add_column("使用次数", justify="right", style="green")
        for tag, count in sorted(metrics["by_tag"].items(), key=lambda x: -x[1])[:10]:
            table_tag.add_row(tag, str(count))
        console.print(table_tag)


@lesson_app.command("deprecate")
def deprecate_lesson(
    workflow_id: str = typer.Argument(..., help="Workflow ID to mark as deprecated"),
    reason: str = typer.Argument(..., help="Reason for deprecation"),
    reviews_file: Path = typer.Option(
        Path("var/reviews/input-reviews.jsonl"),
        "--reviews",
        help="Path to input reviews JSONL file",
    ),
    output: Path = typer.Option(
        None, "--output", help="Output path for deprecation records"
    ),
) -> None:
    """Mark a lesson as deprecated with a reason.

    Example:
        magvasp lesson deprecate FeO-candidate-01 "使用了过时的 U 参数"
    """
    manager = LessonManager(reviews_file)
    output_path = output or reviews_file.parent / "deprecated-lessons.jsonl"

    success = manager.mark_deprecated(workflow_id, reason, output_path)

    if success:
        console.print(
            f"[green]✓[/green] 已标记经验 [cyan]{workflow_id}[/cyan] 为过期"
        )
        console.print(f"   原因: {reason}")
        console.print(f"   记录写入: {output_path}")
    else:
        console.print(
            f"[red]✗[/red] 标记失败: 无法找到 workflow {workflow_id}", err=True
        )
        raise typer.Exit(1)
