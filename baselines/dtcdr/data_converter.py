"""
DTCDR Data Converter

CoNet과 동일한 데이터 변환 로직 사용
"""

from ..conet.data_converter import CoNetDataConverter, CoNetSample


# DTCDR uses same data format as CoNet
DTCDRDataConverter = CoNetDataConverter
DTCDRSample = CoNetSample
