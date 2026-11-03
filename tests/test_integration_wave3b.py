"""Real-model Wave 3b long-audio smoke (set RUN_INTEGRATION=1).

Several unique TTS paragraphs are concatenated with silent gaps into one
4-6 minute WAV, then aligned through the default ASR-anchored window path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.audio_io import duration_s, read_audio, write_audio
from qtts_cli.cli import main

_raw_models = os.environ.get("QTTS_SMOKE_MODEL_DIR")
MODELS = Path(_raw_models).expanduser() if _raw_models else None
TTS_MODEL = MODELS / "Qwen3-TTS-12Hz-1.7B-CustomVoice" if MODELS is not None else None
ASR_MODEL = MODELS / "Qwen3-ASR-0.6B" if MODELS is not None else None
ALIGNER_MODEL = MODELS / "Qwen3-ForcedAligner-0.6B" if MODELS is not None else None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run real-model integration tests",
    ),
    pytest.mark.skipif(
        MODELS is None or not MODELS.is_dir(),
        reason="set QTTS_SMOKE_MODEL_DIR to the local model directory",
    ),
    pytest.mark.skipif(
        any(model is None or not model.is_dir() for model in (TTS_MODEL, ASR_MODEL, ALIGNER_MODEL)),
        reason="TTS/ASR/ForcedAligner models not found under QTTS_SMOKE_MODEL_DIR",
    ),
]

# Unique paragraphs: repeated text would make window-to-text anchors ambiguous.
PARAGRAPHS = [
    "清晨的河面浮着薄雾，岸边的柳树轻轻摇动。早起的渔民收起渔网，木船慢慢驶向石桥。",
    "午后的小巷安静而温暖，墙角的猫在阳光里打盹。卖花的人推着旧车经过，留下一阵淡淡清香。",
    "山顶的风吹动经幡，远处的雪峰在云层间闪光。旅人坐在石阶上喝水，听见寺庙传来钟声。",
    "雨后的稻田泛着新绿，田埂上积着明亮的水洼。白鹭从水面飞起，翅膀掠过远处的村屋。",
    "夜晚的车站灯火通明，广播声在长廊里回荡。有人拖着行李赶来，有人抱着鲜花等待。",
    "老街的茶馆飘出热气，木桌上摆着青瓷小杯。说书人敲响醒木，听众的笑声传到门外。",
    "清晨的菜市场十分热闹，摊主把青菜摆得整整齐齐。顾客提着竹篮挑选，空气里混着泥土气味。",
    "海边的灯塔刚刚亮起，浪花一遍遍拍打礁石。归港的渔船排成一线，桅杆在暮色里摇晃。",
    "深秋的树林铺满落叶，脚踩上去发出细碎声响。松鼠抱着松果跑过，树梢的乌鸦忽然起飞。",
    "冬日的小屋生着炉火，窗玻璃蒙着一层白雾。孩子把手贴在窗上，画出一个弯弯的月亮。",
    "春天的公园开满郁金香，长椅上坐着读书的老人。风筝在天空排成雁阵，笑声从草地传来。",
    "夏夜的院子摆着竹床，葡萄藤下挂着旧灯泡。邻居端来切好的西瓜，蒲扇摇出阵阵凉风。",
    "图书馆的窗边光线柔和，书页翻动时几乎没有声音。管理员整理旧报纸，时钟指向下午四点。",
    "火车站旁的旧书店不大，书架上堆着发黄的画册。老板戴着圆框眼镜，替客人寻找绝版小说。",
    "草原上的帐篷升起炊烟，牧羊犬围着羊群奔跑。远处传来马头琴声，晚霞把天空染成橘红。",
    "江南的石桥横跨小河，桥下乌篷船缓缓划过。洗衣的妇人唱着歌，水声和歌声混在一起。",
    "北方的麦田一望无际，收割机在田里来回穿梭。农人站在树荫下，数着装满粮食的卡车。",
    "高原的湖泊蓝得像宝石，岸边的牦牛低头吃草。风吹过经筒，带来远处雪山的凉意。",
    "城市的清晨地铁繁忙，人们低头看手机和报纸。咖啡店刚开门，第一杯热饮冒着白气。",
    "乡村的午后十分安静，只有蝉声穿过树林。老人坐在门槛上编竹篮，小狗趴在脚边睡觉。",
    "河谷里的果园结满苹果，枝头被果实压得低垂。孩子们提着篮子奔跑，笑声惊起一群麻雀。",
    "雪山脚下的木屋冒着烟，屋檐挂着长长的冰柱。猎人整理皮靴，准备在天亮前出门。",
    "古镇的夜市刚刚开张，灯笼把石板路照得通红。摊位上摆着手工糖人，香味飘向河对岸。",
    "晨雾散去的湖面平静，几只野鸭慢慢游向芦苇。钓鱼的人收起鱼竿，提着水桶沿堤回家。",
    "沿江的步道种着银杏，秋天叶子铺成金色地毯。晨跑的人经过，脚下发出清脆的响声。",
    "山腰的茶园一层层铺开，采茶人背着竹篓弯腰。雾气从谷底升起，把远山遮得朦朦胧胧。",
    "沙漠边缘有一片绿洲，泉水从石缝里缓缓流出。商队在这里歇脚，骆驼低头饮着清水。",
    "渔村的小码头停满木船，绳子在风里轻轻碰响。妇人坐在门前补网，孩子追着海鸥跑远。",
    "森林深处的小路长满苔藓，阳光从树叶间洒下。护林员背着水壶巡逻，记录每一棵老树的编号。",
    "傍晚的校园渐渐安静，教室的灯一盏盏熄灭。学生在操场散步，讨论明天的考试和旅行。",
    "港口边的仓库排列整齐，起重机把集装箱吊上货轮。海风带着咸味，汽笛声在远处回荡。",
    "山谷里的瀑布落下深潭，水雾在阳光下形成彩虹。游客沿着木栈道行走，听见水声越来越近。",
    "旧城区的屋顶连成一片，鸽子在瓦片间来回踱步。远处钟楼敲响，街角的咖啡店飘出香味。",
    "菜园边的篱笆爬满牵牛花，清晨开出紫色的小喇叭。老人提着水壶浇水，蝴蝶停在豆角叶上。",
]

TARGET_SECONDS = 318.0
MAX_SECONDS = 360.0
GAP_SECONDS = 1.0


def _run(tmp_path: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(MODELS),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


@pytest.fixture(scope="module")
def long_audio(tmp_path_factory) -> tuple[Path, str]:
    tmp_path = tmp_path_factory.mktemp("wave3b-integration")
    pieces: list[np.ndarray] = []
    texts: list[str] = []
    sample_rate: int | None = None
    total = 0.0
    for index, paragraph in enumerate(PARAGRAPHS):
        clip = tmp_path / f"clip_{index:02d}.wav"
        rc = _run(
            tmp_path,
            "say",
            paragraph,
            "--voice",
            "builtin:Vivian",
            "--language",
            "Chinese",
            "--out",
            str(clip),
        )
        assert rc == 0, f"TTS failed for paragraph {index}"
        samples, sr = read_audio(clip)
        if sample_rate is None:
            sample_rate = sr
        assert sr == sample_rate
        if pieces:
            pieces.append(np.zeros(int(sample_rate * GAP_SECONDS), dtype=samples.dtype))
            total += GAP_SECONDS
        pieces.append(samples)
        texts.append(paragraph)
        total += duration_s(samples, sr)
        if total >= TARGET_SECONDS:
            break
    assert sample_rate is not None
    assert TARGET_SECONDS <= total <= MAX_SECONDS, (
        f"generated {total:.1f}s of audio; expected {TARGET_SECONDS:.0f}-"
        f"{MAX_SECONDS:.0f}s (extend PARAGRAPHS if needed)"
    )
    output = tmp_path / "long.wav"
    write_audio(output, np.concatenate(pieces), sample_rate, "wav")
    return output, "".join(texts)


def test_align_real_long_audio(capsys, tmp_path: Path, long_audio):
    audio, text = long_audio
    prefix = tmp_path / "aligned"
    recipe = tmp_path / "aligned.recipe.json"
    rc = _run(
        tmp_path,
        "align",
        str(audio),
        "--text",
        text,
        "--language",
        "Chinese",
        "--format",
        "srt,json",
        "--out-prefix",
        str(prefix),
        "--recipe-out",
        str(recipe),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["windowing"]["enabled"] is True
    assert payload["windowing"]["approximate"] is False
    assert payload["windowing"]["asr_model"] == "Qwen3-ASR-0.6B"
    assert len(payload["windowing"]["windows"]) >= 2
    assert payload["coverage"] == 1.0

    document = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
    cues = document["cues"]
    assert cues
    assert "".join(cue["text"] for cue in cues) == text
    starts = [cue["start"] for cue in cues]
    ends = [cue["end"] for cue in cues]
    assert starts == sorted(starts)
    assert ends == sorted(ends)
    assert all(start <= end for start, end in zip(starts, ends))
    assert ends[-1] >= payload["duration_s"] - 5.0
    assert ends[-1] <= payload["duration_s"] + 1e-6

    recipe_document = json.loads(recipe.read_text(encoding="utf-8"))
    assert recipe_document["windowing"] == payload["windowing"]
    print(
        f"\n[wave3b] duration={payload['duration_s']:.1f}s "
        f"windows={len(payload['windowing']['windows'])} cues={len(cues)} "
        f"last_end={ends[-1]:.2f}s"
    )
