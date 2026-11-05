"""Pure benchmark planning, execution and scoring helpers (FR-P2-06).

Nothing in this module imports torch, qwen-tts or qwen-asr.  The model runtime
is injected by :mod:`qtts_cli.commands.bench`, which keeps the timing and
aggregation rules unit-testable and lets ``--dry-run`` stay import-free.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

from .audio_qc import cosine_similarity
from .budget import DEFAULT_MAX_NEW_TOKENS
from .errors import InputError, ModelError
from .plan import SAMPLING_DEFAULTS, build_plan
from .qc import levenshtein, normalize_for_qc, qc_language, text_units
from .utils import peak_rss_bytes, sha256_text

SCHEMA_VERSION = 1
CORPUS_VERSION = "v1"
SEGMENTER_VERSION = 1

# Stable, short sentence corpora.  Each case carries its own language so an
# explicit language filter never has to guess from the script.
SMOKE_CORPUS: tuple[tuple[str, str, str], ...] = (
    (
        "smoke-001",
        "这是一次语音合成性能测试，用来测量本机生成一段完整句子的速度。",
        "Chinese",
    ),
    (
        "smoke-002",
        "This is a speech synthesis benchmark for measuring how fast one complete "
        "sentence is generated on this machine.",
        "English",
    ),
)

BATCH_CORPUS: tuple[tuple[str, str, str], ...] = (
    ("batch-001", "今天天气很好，我们一起去公园散步吧。", "Chinese"),
    ("batch-002", "请把会议纪要发到我的邮箱，谢谢。", "Chinese"),
    ("batch-003", "这家餐厅的牛肉面味道非常好。", "Chinese"),
    ("batch-004", "火车将在十分钟后到达第三站台。", "Chinese"),
    ("batch-005", "记得出门前关掉客厅的灯和空调。", "Chinese"),
    ("batch-006", "孩子正在书房里安静地阅读故事书。", "Chinese"),
    ("batch-007", "周五下午三点我们开一个简短的项目会。", "Chinese"),
    ("batch-008", "雨停以后，街边的树叶显得格外清新。", "Chinese"),
    ("batch-009", "The quick brown fox jumps over the lazy dog.", "English"),
    ("batch-010", "Please send the meeting notes to my inbox.", "English"),
    ("batch-011", "The train will arrive at platform three in ten minutes.", "English"),
    ("batch-012", "Remember to turn off the lights before you leave.", "English"),
    ("batch-013", "She is reading a storybook quietly in the study.", "English"),
    ("batch-014", "We will hold a short project meeting on Friday afternoon.", "English"),
    ("batch-015", "The restaurant serves excellent beef noodles.", "English"),
    ("batch-016", "After the rain, the leaves along the street look fresh.", "English"),
)

LONGFORM_ZH = (
    "清晨的城市还没有完全醒来，街道上只有零星的车辆和早起散步的人。"
    "我沿着河岸慢慢走，看见水面映着淡淡的霞光，远处的桥像一条安静的线。"
    "这样的时刻总让人觉得，时间并不只是钟表上的数字，它还藏在风、光和水声里。"
    "一位老人提着鸟笼从身边经过，笼中的鸟轻轻叫了两声，声音很快被晨风带走。"
    "路边的早餐铺已经开了门，蒸笼里冒出的白气在冷空气里格外明显，"
    "老板娘一边招呼客人，一边把刚出锅的包子放进纸袋。"
    "我买了一杯热豆浆，坐在长椅上，看着上班的人逐渐多起来。"
    "有人低头看手机，有人快步追赶公交车，也有人像我一样，暂时没有急着去什么地方。"
    "城市的声音一层一层地叠起来，先是树叶摩擦的细响，接着是自行车铃声，"
    "然后是远处工地传来的低沉的机器声。"
    "这些声音并不悦耳，却让人清楚地感到，生活正在继续向前。"
    "我想起小时候住在老房子里，每天清晨都能听见邻居开门的吱呀声，"
    "还有楼下卖豆腐的吆喝。那时的街道比现在窄，房子也比现在矮，"
    "可是人与人之间的距离似乎更近一些。"
    "谁家做了好吃的，总会端一碗给隔壁；谁家孩子放学没人接，邻居也会顺手照看。"
    "后来城市越长越高，道路越来越宽，熟悉的面孔却慢慢分散到不同的地方。"
    "我并不是想说过去一定更好，只是有些东西在变化中悄悄消失了，"
    "等我们想起来的时候，已经很难找回原来的样子。"
    "豆浆喝到一半，太阳从楼群后面升起来，光线落在河面上，碎成一片温暖的金色。"
    "我把杯子丢进分类垃圾桶，起身继续往前走。"
    "前方的路口有人在等红灯，也有人趁着车少快步穿过。"
    "生活大概就是这样，一边告别，一边出发，一边在普通的早晨里寻找继续的理由。"
    "风从河面吹来，带着一点潮湿的凉意。"
    "我忽然觉得，所谓城市，不只是建筑和道路的集合，"
    "它更是无数普通人每天重复的脚步声、说话声和沉默。"
    "这些细小的事情加在一起，才构成了我们共同生活的地方。"
    "走到桥边时，我停下来回头看了一眼。"
    "早餐铺的白气还在升，老人已经走远，河岸上的行人换了一批又一批。"
    "我知道明天清晨，这里仍会发生相似的事情，而每一次相似之中，又会有新的不同。"
    "也许正因为这样，平凡的日常才值得被认真记录。"
    "继续向前走，街道逐渐变得热闹。"
    "修鞋摊的师傅把工具箱摆在门口，旁边的报刊亭刚刚拉起卷帘。"
    "一位母亲牵着孩子的手，孩子一边走一边问今天会不会下雨，"
    "母亲抬头看了看天，说云还不算厚，应该能等到放学。"
    "这样的对话每天都在发生，听起来普通，却让人感到踏实。"
    "我经过一家旧书店，玻璃窗里堆着发黄的杂志和翻旧的小说，"
    "店主坐在柜台后面，借着台灯的光慢慢整理书脊。"
    "他没有抬头，也没有招揽客人，仿佛并不在意今天能卖出几本。"
    "在这个什么都讲求效率的时代，这样的慢显得有点固执，却也让人安心。"
    "再往前走，是一所小学。校门还没有打开，几个孩子已经在门口等候，"
    "书包比他们的背还要宽，笑声却一点也不显得沉重。"
    "保安师傅拿着保温杯站在一旁，偶尔提醒他们不要站到马路上。"
    "阳光越过围墙，落在操场边那排梧桐树上，叶子被照得透亮。"
    "我想，很多年以后，这些孩子也许会忘记某一次考试的成绩，"
    "却可能记得某个早晨的风、某棵树影，以及身边同学说过的一句玩笑。"
    "人的记忆总是这样，不按照重要程度排列，而是被气味、声音和光线悄悄决定。"
    "走到公司楼下时，我看了一眼时间，距离上班还有十几分钟。"
    "于是我在门口的花坛边站了一会儿，把剩下的豆浆喝完。"
    "楼上有人在擦玻璃，水顺着墙面慢慢流下来，在阳光里闪了一下。"
    "清洁工推着车从旁边经过，向我点了点头，我也朝他点头。"
    "我们并不认识，但在这个普通的早晨，彼此的出现让城市显得更有温度。"
    "电梯门打开又合上，新的一天正式开始。"
    "我知道接下来会有会议、邮件和做不完的事情，"
    "但刚才那段不紧不慢的路，会像一枚小小的书签，夹在忙碌的页码之间。"
    "它提醒我，生活不只在目标达成的那一刻发生，"
    "也在每一次呼吸、每一段步行和每一个不经意的相遇里发生。"
)

LONGFORM_EN = (
    "The city is still half asleep when I leave the house. A few cars pass along "
    "the riverside road, and a handful of early walkers move under the trees. "
    "The water carries the pale light of dawn, and the bridge in the distance "
    "looks like a quiet line drawn across the morning. "
    "At moments like this, time feels larger than the numbers on a clock. "
    "It lives in the wind, in the changing light, and in the small sounds that "
    "are usually buried under the noise of the day. "
    "An old man walks past carrying a bird cage. The bird calls twice, and the "
    "sound is carried away almost before I notice it. "
    "Near the corner, a breakfast shop has already opened. Steam rises from the "
    "bamboo baskets and turns white in the cool air. The owner greets customers "
    "while folding fresh buns into paper bags. "
    "I buy a cup of warm soy milk and sit on a bench to watch the street fill up. "
    "Some people study their phones, some hurry toward a bus, and a few, like me, "
    "are not in a rush to be anywhere in particular. "
    "The sounds of the city gather in layers: leaves brushing together, bicycle "
    "bells, and the low hum of machinery from a construction site farther away. "
    "None of it is especially beautiful, but together it makes it clear that life "
    "is continuing. "
    "I remember the old neighborhood where I grew up. Every morning I could hear "
    "a neighbor's door creak open and a man downstairs calling out that fresh tofu "
    "had arrived. The streets were narrower then, and the houses were lower, but "
    "the distance between people seemed shorter. "
    "If a family cooked something special, a bowl would be carried next door. If a "
    "child came home from school and nobody was there, a neighbor would keep an eye "
    "on them until someone returned. "
    "Later the city grew taller and the roads grew wider, and the familiar faces "
    "spread out into different parts of town. I am not saying that the past was "
    "better. I only mean that some things disappeared so gradually that we did not "
    "notice until they were already hard to find. "
    "Halfway through the soy milk, the sun rises above the buildings. Its light "
    "falls across the river and breaks into a warm gold. I throw the empty cup into "
    "the recycling bin and keep walking. "
    "At the next crossing, some people wait for the light while others hurry across "
    "when the road is briefly clear. This is probably what life is: saying goodbye "
    "and setting out again, looking for a reason to continue inside an ordinary "
    "morning. "
    "The wind coming off the river carries a little dampness. It occurs to me that "
    "a city is not only a collection of buildings and roads. It is also the repeated "
    "footsteps, conversations, and silences of ordinary people. Only when all those "
    "small things are added together does a shared place begin to exist. "
    "At the bridge I stop and look back. The steam is still rising from the breakfast "
    "shop, the old man is out of sight, and a new group of walkers has taken the "
    "place of the last. Tomorrow morning the same scenes will return, and within "
    "each repetition there will be something slightly different. Perhaps that is why "
    "ordinary days deserve to be recorded with care."
)


@dataclass(frozen=True)
class BenchCase:
    """One planned generation; timing fields are filled by :func:`execute`."""

    id: str
    text: str
    language: str
    instruct: Optional[str] = None
    seed: Optional[int] = None
    max_new_tokens: Optional[int] = None
    suite_index: int = 0

    @property
    def text_sha256(self) -> str:
        return sha256_text(self.text)


@dataclass
class BenchExecution:
    metrics: dict[str, Any]
    runs: list[dict[str, Any]]
    cases: list[dict[str, Any]]
    last_run_audio: list[tuple[BenchCase, Any, int]]
    device_memory_mb: Optional[float] = None
    device_memory_kind: Optional[str] = None


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: Any, digits: int) -> Optional[float]:
    number = _finite(value)
    return None if number is None else round(number, digits)


def corpus_id(suite: str) -> str:
    return f"{suite}.{CORPUS_VERSION}"


def smoke_cases(language: str = "auto") -> list[BenchCase]:
    cases: list[BenchCase] = []
    for index, (case_id, text, case_language) in enumerate(SMOKE_CORPUS, 1):
        if language != "auto" and case_language.lower() != language.lower():
            continue
        cases.append(
            BenchCase(
                id=case_id,
                text=text,
                language=case_language,
                suite_index=index,
            )
        )
    return cases


def batch_cases(language: str, count: int) -> list[BenchCase]:
    pool = [
        (text, case_language)
        for _, text, case_language in BATCH_CORPUS
        if language == "auto" or case_language.lower() == language.lower()
    ]
    if not pool:
        return []
    cases: list[BenchCase] = []
    for index in range(count):
        text, case_language = pool[index % len(pool)]
        cases.append(
            BenchCase(
                id=f"batch-{index + 1:03d}",
                text=text,
                language=case_language,
                suite_index=index + 1,
            )
        )
    return cases


def builtin_longform_text(language: str) -> str:
    return LONGFORM_EN if language.lower() == "english" else LONGFORM_ZH


def longform_cases(
    cfg,
    *,
    text: str,
    voice_ref: str,
    language: str,
    models: Iterable[Any],
    store: Any,
    model_override: Optional[str] = None,
) -> tuple[list[BenchCase], Any]:
    plan = build_plan(
        cfg,
        text=text,
        voice_ref=voice_ref,
        language=language,
        models=models,
        store=store,
        model_override=model_override,
    )
    cases = [
        BenchCase(
            id=f"longform-{segment.id}",
            text=segment.text,
            language=segment.language,
            instruct=segment.instruct,
            seed=segment.seed,
            max_new_tokens=segment.max_new_tokens,
            suite_index=index,
        )
        for index, segment in enumerate(plan.segments, 1)
    ]
    return cases, plan


def corpus_payload(
    suite: str,
    *,
    source: str,
    language: str,
    cases: Sequence[BenchCase],
    normalization_version: Optional[int] = None,
    segmenter_version: Optional[int] = None,
) -> dict[str, Any]:
    texts = "\n".join(case.text for case in cases)
    return {
        "id": corpus_id(suite),
        "source": source,
        "language": language,
        "cases": len(cases),
        "text_sha256": sha256_text(texts),
        "normalization_version": normalization_version,
        "segmenter_version": segmenter_version,
    }


def case_plan_payload(
    case: BenchCase,
    *,
    default_seed: Optional[int] = None,
    default_max_new_tokens: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "id": case.id,
        "run": None,
        "suite_index": case.suite_index,
        "text": case.text,
        "text_sha256": case.text_sha256,
        "language": case.language,
        "instruct": case.instruct,
        "seed": case.seed if case.seed is not None else default_seed,
        "max_new_tokens": (
            case.max_new_tokens
            if case.max_new_tokens is not None
            else default_max_new_tokens
        ),
        "audio_duration_s": None,
        "generate_ms": None,
        "rtf": None,
    }


def _sampling_for(case: BenchCase, sampling: Optional[dict[str, Any]]) -> dict[str, Any]:
    values = dict(SAMPLING_DEFAULTS)
    values["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
    if sampling:
        values.update(sampling)
    if case.max_new_tokens is not None:
        values["max_new_tokens"] = int(case.max_new_tokens)
    return values


def execute(
    engine,
    cases: Sequence[BenchCase],
    runs: int,
    *,
    mode: str,
    speaker: Optional[str] = None,
    prompt_items: Optional[list[Any]] = None,
    sampling: Optional[dict[str, Any]] = None,
    seed: int = 1234,
    clock: Callable[[], float] = time.perf_counter,
    peak_rss: Callable[[], int] = peak_rss_bytes,
    baseline_rss: Optional[int] = None,
    device_memory: Optional[Callable[[], tuple[Optional[float], Optional[str]]]] = None,
) -> BenchExecution:
    """Run every case ``runs`` times and aggregate external wall-clock metrics."""
    if not cases:
        raise InputError("benchmark has no cases to run")
    if runs < 1:
        raise InputError("benchmark runs must be positive")

    base_rss = int(baseline_rss) if baseline_rss is not None else int(peak_rss())
    load_start = clock()
    engine.load()
    load_end = clock()
    # §A: every timing metric uses the external, unrounded clock.  An engine's
    # self-reported load_ms starts after runtime import and would undercount
    # cold start, so it is deliberately ignored here.
    load_ms = max(0.0, (load_end - load_start) * 1000.0)

    run_records: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    run_generate_s: list[float] = []
    run_audio_s: list[float] = []
    first_generate_ms: list[float] = []
    last_run_audio: list[tuple[BenchCase, Any, int]] = []

    for run_index in range(1, runs + 1):
        run_start = clock()
        generated_s = 0.0
        audio_s = 0.0
        first_ms: Optional[float] = None
        collected: list[tuple[BenchCase, Any, int]] = []

        for case in cases:
            case_sampling = _sampling_for(case, sampling)
            case_seed = int(case.seed if case.seed is not None else seed)
            generate_start = clock()
            audio, sr, info = engine.generate(
                text=case.text,
                language=case.language,
                mode=mode,
                speaker=speaker,
                instruct=case.instruct,
                prompt_items=prompt_items,
                sampling=case_sampling,
                seed=case_seed,
            )
            generate_wall = max(0.0, clock() - generate_start)
            if first_ms is None:
                first_ms = generate_wall * 1000.0

            duration = float(len(audio)) / float(sr) if int(sr) > 0 else 0.0
            duration = max(0.0, duration)
            generated_s += generate_wall
            audio_s += duration
            collected.append((case, audio, int(sr)))
            case_records.append(
                {
                    "id": case.id,
                    "run": run_index,
                    "suite_index": case.suite_index,
                    "text": case.text,
                    "text_sha256": case.text_sha256,
                    "language": case.language,
                    "instruct": case.instruct,
                    "seed": case_seed,
                    "max_new_tokens": int(case_sampling["max_new_tokens"]),
                    "audio_duration_s": round(duration, 3),
                    "generate_ms": _round(info.get("generate_ms"), 1),
                    "rtf": (
                        round(generate_wall / duration, 3)
                        if duration > 0.0
                        else None
                    ),
                }
            )

        run_end = clock()
        run_wall = max(0.0, run_end - run_start)
        run_rtf = generated_s / audio_s if audio_s > 0.0 else None
        peak = int(peak_rss())
        run_generate_s.append(generated_s)
        run_audio_s.append(audio_s)
        first_generate_ms.append(float(first_ms or 0.0))
        last_run_audio = collected
        run_records.append(
            {
                "index": run_index,
                "first_segment_ms": round(float(first_ms or 0.0), 1),
                "wall_s": round(run_wall, 3),
                "generate_ms": round(generated_s * 1000.0, 1),
                "audio_s": round(audio_s, 3),
                "rtf": None if run_rtf is None else round(run_rtf, 3),
                "segments": len(cases),
                "process_peak_rss_bytes": peak,
                "cer": None,
                "speaker_similarity": None,
            }
        )

    wall_s = max(0.0, clock() - load_start)
    total_generate_s = sum(run_generate_s)
    total_audio_s = sum(run_audio_s)
    rtf_values = [
        generated / audio for generated, audio in zip(run_generate_s, run_audio_s) if audio > 0.0
    ]
    close_rtf = len(rtf_values) == len(run_generate_s) and bool(rtf_values)
    metrics_rtf = statistics.median(rtf_values) if close_rtf else None
    wall_rtf = wall_s / total_audio_s if total_audio_s > 0.0 else None
    segments_per_run = len(cases)
    segments_total = segments_per_run * runs
    final_peak = int(peak_rss())

    device_mb: Optional[float] = None
    device_kind: Optional[str] = None
    if device_memory is not None:
        try:
            device_mb, device_kind = device_memory()
            device_mb = _round(device_mb, 1)
        except Exception:
            device_mb, device_kind = None, None

    metrics: dict[str, Any] = {
        "load_ms": _round(load_ms, 1),
        "cold_first_segment_ms": _round(load_ms + first_generate_ms[0], 1),
        "warm_first_segment_ms": _round(statistics.median(first_generate_ms), 1),
        "wall_s": round(wall_s, 3),
        "generate_s": round(total_generate_s, 3),
        "audio_s": round(total_audio_s, 3),
        "rtf": None if metrics_rtf is None else round(metrics_rtf, 3),
        "wall_rtf": None if wall_rtf is None else round(wall_rtf, 3),
        "segments_per_run": segments_per_run,
        "segments_total": segments_total,
        "segments_per_s": (
            round(segments_total / total_generate_s, 3)
            if total_generate_s > 0.0
            else None
        ),
        "process_peak_rss_bytes": final_peak,
        "peak_rss_delta_bytes": max(0, final_peak - base_rss),
        "device_memory_mb": device_mb,
        "device_memory_kind": device_kind,
        "cer": None,
        "cer_distance": None,
        "cer_units": None,
        "speaker_similarity": None,
        "speaker_segments_scored": None,
        "speaker_segments_total": None,
    }
    return BenchExecution(
        metrics=metrics,
        runs=run_records,
        cases=case_records,
        last_run_audio=last_run_audio,
        device_memory_mb=device_mb,
        device_memory_kind=device_kind,
    )


def score_cer(asr, audio_items: Sequence[tuple[BenchCase, Any, int]]) -> dict[str, Any]:
    """Accumulate character edit distance over the last run's cases."""
    distance = 0
    reference_units = 0
    for case, audio, sr in audio_items:
        language = qc_language(case.language, case.text)
        result = asr.transcribe_samples(audio, int(sr), language=language)
        if not isinstance(result, dict):
            raise ModelError("ASR returned no single transcription result")
        transcript = result.get("text")
        if not isinstance(transcript, str):
            raise ModelError("ASR returned no transcription text")
        reference = text_units(normalize_for_qc(case.text, language), "cer")
        hypothesis = text_units(normalize_for_qc(transcript, language), "cer")
        distance += levenshtein(reference, hypothesis)
        reference_units += len(reference)
    ratio = distance / reference_units if reference_units > 0 else None
    return {
        "cer": None if ratio is None else round(ratio, 6),
        "cer_distance": distance,
        "cer_units": reference_units,
    }


def mean_cosine_similarity(
    reference: Any, embeddings: Sequence[Any]
) -> tuple[Optional[float], int]:
    """Mean cosine; any empty/zero/mismatched vector makes the result null.

    Every segment is still scored so ``segments_scored`` is truthful even when
    one segment makes the aggregate unavailable.
    """
    values: list[float] = []
    invalid = False
    for embedding in embeddings:
        value = cosine_similarity(reference, embedding)
        if value is None:
            invalid = True
            continue
        values.append(float(value))
    if invalid or not values:
        return None, len(values)
    return round(statistics.fmean(values), 4), len(values)


def score_speaker(
    embedder,
    reference_embedding: Any,
    audio_items: Sequence[tuple[BenchCase, Any, int]],
    *,
    reference_audio: Optional[tuple[Any, int]] = None,
) -> tuple[Optional[float], int, int]:
    """Score the last run's raw generated audio against one reference embedding."""
    if reference_embedding is None:
        if reference_audio is None:
            raise InputError("speaker similarity has no reference embedding or audio")
        samples, sr = reference_audio
        reference_embedding = embedder.extract_speaker_embedding(samples, int(sr))
    embeddings = [
        embedder.extract_speaker_embedding(audio, int(sr))
        for _, audio, sr in audio_items
    ]
    similarity, scored = mean_cosine_similarity(reference_embedding, embeddings)
    return similarity, scored, len(audio_items)
