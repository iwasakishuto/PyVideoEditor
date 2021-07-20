## Sample Data

The following data is used as a sample file

|Name|Reference|
|:-:|:-|
|`file_example_MP4_1280_10MG.mp4`|[File Examples](https://file-examples.com/)|
|`file_example_PNG_500kB.png`|[File Examples](https://file-examples.com/)|
|`fonts/pokemon-font/fonts/`|[pokemon-font](https://github.com/PascalPixel/pokemon-font/tree/60280120447da9de4f0f28ceaacff144642bb16a)|

```python
from veditor.utils import SampleData
datasets = SampleData()
assert hasattr(datasets, 'IMAGE_PATH')
assert hasattr(datasets, 'VIDEO_PATH')
assert hasattr(datasets, 'AUDIO_PATH')
assert hasattr(datasets, 'FONT_POKEFONT_PATH')
```