# 官方天基数据的 Gaia 参考星表

`rst19-gaia-dr3.csv` 是从 CDS VizieR 的 Gaia DR3 表 `I/355/gaiadr3` 获取的
1,860 行参考星数据，2026-09-23 查询，中心 ICRS (129.533548°, −1.845372°)，
半径 3°，G < 12。只保存 Source、RA_ICRS、DE_ICRS、Gmag 四列，并重命名为
`source_id,ra_deg,dec_deg,gmag`。source_id 保留为十进制字符串，不经浮点转换。

来源：[Gaia DR3 数据说明](https://cdsarc.cds.unistra.fr/viz-bin/cat/I/355)，
[固定查询](https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=I%2F355%2Fgaiadr3&-c=129.533548%20-1.845372&-c.rd=3&-out=Source%2CRA_ICRS%2CDE_ICRS%2CGmag&-out.max=10000&Gmag=%3C12)。
数据引用：Gaia Collaboration (2022)，DOI [10.26093/cds/vizier.1355](https://doi.org/10.26093/cds/vizier.1355)。
[VizieR 数据使用与引用说明](https://cds.unistra.fr/vizier-org/licences_vizier.html)。

CSV SHA-256：`653e31b38411a409a84083c299b211e67f512d24bb5f84295e222c896e4f9d62`。
裁判可直接使用包内 CSV 离线复现，无需账户或网络。它是参考数据，不是目标真值，
只在独立检测完成后用于星表匹配与定标。

Gaia 坐标历元为 2016.0，本次未引入自行传播；高自行、混叠、变星和相机响应差异
可能形成离群值。程序执行一对一位置配对及稳健剪裁，并记录拟合使用的每一颗星。
相机滤光响应未知，因此输出为 **Gaia G 参照的近似星等**，不能称为标准 V 星等，
零点拟合散度也不包含全部系统误差。不同天区应另提供覆盖该天区的星表，程序不会
把本文件的零点直接套给其他输入。
