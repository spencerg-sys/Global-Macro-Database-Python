from __future__ import annotations

from .afdb import download_afdb
from .ameco import download_ameco
from .bceao import download_bceao
from .bis_cbrate import download_bis_cbrate
from .bis_cpi import download_bis_cpi
from .bis_hpi import download_bis_hpi
from .bis_reer import download_bis_reer
from .bis_usdfx import download_bis_usdfx
from .eus import download_eus
from .fred import download_fred
from .franc_zone import download_franc_zone
from .idcm import download_idcm
from .imf_gfs import download_imf_gfs
from .imf_ifs import download_imf_ifs
from .imf_mfs import download_imf_mfs
from .imf_weo import download_imf_weo
from .oecd_eo import download_oecd_eo
from .oecd_hpi import download_oecd_hpi
from .oecd_kei import download_oecd_kei
from .oecd_mei import download_oecd_mei
from .oecd_qna import download_oecd_qna
from .oecd_rev import download_oecd_rev
from .un import download_un
from .wdi import download_wdi

__all__ = [
    "download_afdb",
    "download_ameco",
    "download_bceao",
    "download_bis_cbrate",
    "download_bis_cpi",
    "download_bis_hpi",
    "download_bis_reer",
    "download_bis_usdfx",
    "download_eus",
    "download_fred",
    "download_franc_zone",
    "download_idcm",
    "download_imf_gfs",
    "download_imf_ifs",
    "download_imf_mfs",
    "download_imf_weo",
    "download_oecd_eo",
    "download_oecd_hpi",
    "download_oecd_kei",
    "download_oecd_mei",
    "download_oecd_qna",
    "download_oecd_rev",
    "download_un",
    "download_wdi",
]
