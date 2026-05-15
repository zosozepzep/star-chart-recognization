import shutil
import subprocess
from pathlib import Path

from .cross_match import read_catalog


DEFAULT_SEXTRACTOR_CONFIG_DIR = Path(__file__).resolve().parent / "sextractor_config"


def find_sextractor_command():
    """
    查找当前环境里的 SExtractor 命令。

    不同系统安装后命令名可能不完全一样：
    - Dockerfile 里安装的是 source-extractor
    - 有些环境里可能叫 sextractor
    - 老版本也可能叫 sex
    """
    for command in ["source-extractor", "sextractor", "sex"]:
        if shutil.which(command) is not None:
            return command

    return None


def run_sextractor_on_directory(
    image_dir,
    output_dir,
    config_dir=None,
    max_frames=None,
    frame_names=None,
):
    """
    对一个目录里的所有 FITS 图像批量运行 SExtractor。

    输入：
    - image_dir: FITS 图像目录，例如 data/images
    - output_dir: SExtractor 输出目录，例如 output/benchmark/sextractor
    - config_dir: SExtractor 配置目录。
      如果不传，就默认使用 src/benchmark/sextractor_config。
    - max_frames: 只处理前 N 帧。用于快速验证流程，正式评测时可以不传。
    - frame_names: 只处理指定文件名列表。用于和已有 pipeline detection 对齐。

    输出：
    - output_dir/catalogs/*.cat: SExtractor 原始 catalog
    - output_dir/catalogs/*.csv: 转换后的 CSV，方便 benchmark 和人工检查
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    if config_dir is None:
        config_dir = DEFAULT_SEXTRACTOR_CONFIG_DIR
    else:
        config_dir = Path(config_dir)
    catalog_dir = output_dir / "catalogs"

    sextractor_command = find_sextractor_command()
    if sextractor_command is None:
        raise RuntimeError(
            "SExtractor command not found. "
            "Please run inside Docker or install source-extractor first."
        )

    check_sextractor_config(config_dir)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    fits_files = find_fits_files(image_dir)
    fits_files = filter_fits_files(fits_files, frame_names=frame_names, max_frames=max_frames)

    if not fits_files:
        raise RuntimeError(f"No FITS files found in {image_dir}")

    generated_catalogs = []

    for fits_path in fits_files:
        cat_path = catalog_dir / f"{fits_path.stem}.cat"
        csv_path = catalog_dir / f"{fits_path.stem}.csv"

        run_sextractor_single_file(
            sextractor_command=sextractor_command,
            fits_path=fits_path,
            cat_path=cat_path,
            config_dir=config_dir,
        )

        # 把 .cat 转成 .csv。
        # .cat 是 SExtractor 的原始输出，CSV 更适合后续 benchmark 和手动查看。
        df = read_catalog(cat_path)
        df.to_csv(csv_path, index=False)
        generated_catalogs.append(csv_path)

    return {
        "catalog_dir": catalog_dir,
        "catalogs": generated_catalogs,
    }


def run_sextractor_single_file(sextractor_command, fits_path, cat_path, config_dir):
    """
    对单个 FITS 文件运行 SExtractor。

    这里显式传入 PARAMETERS_NAME / FILTER_NAME / STARNNW_NAME。
    原因是 default.sex 里写的是相对路径，如果工作目录变化，SE 可能找不到配置文件。
    """
    fits_path = Path(fits_path)
    cat_path = Path(cat_path)
    config_dir = Path(config_dir)

    command = [
        sextractor_command,
        str(fits_path),
        "-c",
        str(config_dir / "default.sex"),
        "-CATALOG_NAME",
        str(cat_path),
        "-PARAMETERS_NAME",
        str(config_dir / "default.param"),
        "-FILTER_NAME",
        str(config_dir / "default.conv"),
        "-STARNNW_NAME",
        str(config_dir / "default.nnw"),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "SExtractor failed on "
            f"{fits_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def find_fits_files(image_dir):
    """
    查找 FITS 文件，并按文件名排序。

    这里不负责按 FITS Header 时间排序；benchmark 只需要逐帧同名匹配，
    所以稳定的文件名排序已经足够。
    """
    image_dir = Path(image_dir)
    files = []
    files.extend(image_dir.glob("*.fit"))
    files.extend(image_dir.glob("*.fits"))
    files.extend(image_dir.glob("*.FIT"))
    files.extend(image_dir.glob("*.FITS"))
    return sorted(files)


def filter_fits_files(fits_files, frame_names=None, max_frames=None):
    """
    对 FITS 文件列表做可选过滤。

    这里主要服务于快速验证：
    - frame_names 可以指定只跑哪些帧
    - max_frames 可以只跑排序后的前 N 帧
    """
    if frame_names is not None:
        allowed_names = set(str(name) for name in frame_names)
        fits_files = [path for path in fits_files if path.name in allowed_names]

    if max_frames is not None:
        fits_files = fits_files[:max_frames]

    return fits_files


def check_sextractor_config(config_dir):
    """
    检查运行 SExtractor 必需的配置文件是否存在。
    """
    required_files = [
        "default.sex",
        "default.param",
        "default.conv",
        "default.nnw",
    ]

    missing_files = []
    for filename in required_files:
        path = Path(config_dir) / filename
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "Missing SExtractor config files: " + ", ".join(missing_files)
        )
