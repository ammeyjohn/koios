---
title: 基于Tensorflow的比赛成绩自动识别服务
date: 2019-09-27 10:21:47
tags: tensorflow,识别
---

## 简介

这个话题来源于我自己参与组织亚太口琴节比赛过程中产生的。在亚太口琴节比赛时，每天约有20~30场比赛，每场比赛约有50~200人参加角逐，比赛后会有专门的成绩录入人员将7位评委的成绩录入到赛事管理系统中。由于当天就需要发布比赛成绩、名次并准备决赛名单，需要把所有成绩录入进系统，花费了大量的人力和时间。因此萌生了是不是可以通过提取并自动识别照片中的成绩，这样免去了大量人工录入的时间。 因此就在业余时间开发了这套成绩自动识别服务，和赛事管理系统相结合，希望在2020年亚太口琴节中得到充分应用，并取得好的效果。借此机会进行总结并与大家分享。 在本场Chat中，您会了解到如下内容：

- 使用OpenCV对图片进行校正以及内容提取；
- 使用PyTesseract识别印刷体文本；
- 使用Tensorflow构建手写读数识别模型用于识别比赛成绩；
- 使用Flask Api实现Restful服务，上传图片，成绩识别；

## 正文

这个话题来源于我自己参与组织亚太口琴节比赛过程中产生的。在亚太口琴节比赛时，每天约有20~30场比赛，每场比赛约有50~200人参加角逐，每场比赛都会有7名评委为选手打分，每场比赛后会有专门的成绩录入人员将评委的成绩录入到赛事管理系统中，并由复核人员对成绩进行最后的审查。由于很多比赛有初赛和决赛，初赛的当天就需要公布比赛成绩、名词以及能够进入决赛的选手名单，所以需要把当天比赛的所有成绩都录入进系统，这个过程需要花费了大量的人力和时间。成绩录入这个过程无疑是口琴节是否能够成功的举办的一大影响因素，也成为了整个口琴节中最大的挑战。

由此便萌生了是不是可以通过图像处理和机器学习相结合的方式，自动从照片中提取的成绩，这样即免去了大量人工录入的时间，也可以提高成绩录入的准确度。因此从自己的这个想法展开，结合自己在机器学习方面的研究，开发了一个可以自动识别比赛成绩的识别服务，与赛事管理系统相结合，可以上传评委评分表，自动从评分表中识别成绩，录入到赛事管理系统的成绩录入界面中，待审核通过后保存成绩。

### 目录结构

- 需求确定
- 准备工作
- 识别流程
- 图片校正和表格提取
- 表格行提取
- 表格单元格提取
- 印刷体数字的识别
- 手写数字提取与切分
- 手写数字模型的构建和训练
- 成绩识别
- 成绩识别服务的构建
- 总结

### 需求确定

1. **图片的来源**

图片可以通过手机拍摄照片后，通过赛事管理系统上传并进行成绩识别。因此赛事管理系统需要提供一个专用页面来处理图片上传、识别、审核。

2. **图片中评分表的样式**

评分表格式是固定的，考虑到每场比赛人数多少的因素，表格的布局会有所差异。评分表分为2页，第一页包括页标题、场次描述和选手评分区域；第二页少了标题、场次描述部分，如下图所示。

{% asset_img 2.png 评分表样式 %}

3. **照片中评分表的校正**

考虑到比赛的时候，工作人员大部分都是场地组织的志愿者，现场拍摄照片的环境复杂多变，拍摄过程中光照强度、倾斜角度、旋转角度都会有所差异，纸张也会有不同程度的折角或者卷边的问题，而且不同手机摄像头所拍摄的分辨率也会有所差异，因此需要考虑对图片进行标准化的处理。

4. **手写数字的识别**

评委来自于不同国家，不同文化、不同书写习惯使评委书写的数字风格迥异，书写的数字的可识别性也大相径庭。如下图所示：

{% asset_img 3.png 不同书写风格的成绩 %}

使用机器学习领域十分有名的MNIST数据集进行训练，由于其数据量少（训练集60000，测试集10000），识别的效果并不理想。因此我使用由Facebook公开的[qmnist](https://github.com/facebookresearch/qmnist)(https://github.com/facebookresearch/qmnist)数据集，其包含402953张手写数字的图片。使用qmnist数据集对模型进行训练后，可以得到不错的识别率。

5. **印刷体数字的识别**

比赛序号与该场次参赛者是一一对应的，在成绩识别后需要根据序号保存对应的成绩，而由于选手可能弃权、换组等原因，序号可能不连续，因此也需要有一个识别算法来识别序号。服务中使用了PyTesseract来对序号进行识别。

6. **识别性能**

识别算法的性能在整个过程中并没有特别考虑，与人工输入相比较，通过机器识别后录入的速度有着质的飞跃，因此可以接受略长的识别性能。

7. **识别准确率**

考虑到很多特殊情况，如成绩被涂改、未填写在规定区域、数字过于潦草等，总体识别准确率考虑不低于90%。

### 准备工作

准备开发前需要先准备一下开发环境，开发工具我使用了VS Code安装了Python的插件，也可以使用社区版的PyCharm，社区版的功能应该是够用的。

另外强烈推荐在开发前安装Anaconda工具，可以免去后面很多库的安装和管理工作。相关库以及版本如下表所示：

- Python 3.7
- numpy 1.16.2
- tensorflow 1.14.0
- matplotlib 3.0.3
- pytorch (qmnist需要该库）1.0.1

可以通过以下命令安装这些组件：

```bash
conda install numpy
conda install matploglib
conda install pytorch
conda install tensorflow
## 如果使用GPU，可以安装gpu版本的Tensorflow
conda install tensorflow-gpu
```

### 识别流程

```flow
st=>start: 开始
op_imread=>inputoutput: 读取图片
op_norm=>subroutine: 图片标准化并提取表格区域
op_table=>subroutine: 识别表格行
cond=>condition: 是否为最后一行
op_row_norm=>subroutine: 行标准化
op_cell=>subroutine: 提取序号列和成绩列
op_id_recognize=>operation: 识别序号

op_score_cut=>subroutine: 成绩切分
op_score_recognize=>subroutine: 识别成绩
ed=>end: 结束

st->op_imread->op_norm->op_table->cond
cond(no, right)->op_row_norm->op_cell(right)->op_id_recognize(right)->cond
cond(yes)->op_score_cut->op_score_recognize->ed
```

从流程图中可以看到，整个识别过程需要经历四个关键过程，表格提取、行提取、列提取、数字识别。

- 表格的定位和提取，从原始图片中将A4纸进行标准化，校正表格的角度，从图片中提取表格区域；完成后的对比图如下，左图为原始的图片，右图为校正后的图片。

{% asset_img 1.jpg 表格提取结果 %}

- 通过表格中的行与行之间的线，提取表格中的每一行；对行进行角度校正；
- 通过表格中的列于列之间的线，提取“序号列”和“评分列”；
- 使用Tesseract-OCR识别序号数字；
- 识别成绩；如果成绩写的比较分开，可以直接提取到三个数字并分别进行识别；如果成绩书写时有连笔，那么需要先切分成绩，然后再依次识别。

### 图片校正和表格提取

图片校正是整个识别过程中十分重要的一步，如果无法准确地将表格区域识别出来，那么可能会导致后续表格行提取不正确，从而影响整个识别的准确程度。图片校正前需要先对图片进行预处理，

- 将图片转换为灰度；
- 对图片进行中值滤波，过滤图片中的噪声；
- 提取图片的边缘信息，用于后续表格边框的提取；

```python
# 转换为灰度
gray_img = utils.to_gray(img_copied)
# 中值滤波 过滤噪声，保留边缘信息
gray_img = cv2.medianBlur(gray_img, 5) 
# Canny算子求得图像边缘
edges_img = cv2.Canny(gray_img, 50, 150, apertureSize = 3)

# 定义一个5×5的十字形状的结构
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
# 重复膨胀 迭代5次
edges_img = cv2.dilate(edges_img, kernel, iterations=5)
```

由于表格线存在一定的宽度，从Canny中提取的图像边缘的图片存在间隙，因此需要将图片进行一次膨胀操作，使直线都连成一块，方便后续识别边框。处理后的图片如图所示：

{% asset_img 4.png Canny边缘提取 %} 
{% asset_img 5.png 边缘膨胀 %}

从图片中提取表格区域。这部分比较简单，OpenCV自带的findContours方法就可以找到图片中的轮廓。由于我们只关系表格周围的轮廓，因此使用RETR_EXTERNAL参数，只范围最外层的轮廓，所有子轮廓都将被忽略。

```python
# 寻找轮廓
contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 获取面积最大的contour
cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))
```

{% asset_img 6.jpg 表格区域识别和提取 %}

```python
# 多变形近似
epsilon = 0.1 * cv2.arcLength(cnt, True)
# 返回点集，格式与contours相同
approx = cv2.approxPolyDP(cnt, epsilon, True)
```

仔细观察生成边框的边缘，你会发现边缘并不是平整的，会有间隙；而且如果把cnt打印出来的话，可以看到这个蓝色的框其实并不是一个矩形，而是通过很多的坐标点描绘出来的，因此我们需要通过轮廓近似的方法，从这些点集合中生成一个标准的矩形。其中epsilon参数越小，会越近原始的轮廓，越大，那么会越接近直线。

```python
def calc_distance(pt0, pt1):
    """
    计算两点之间的距离
    :param pt0: 点0
    :param pt1: 点1
    :return: 返回两点之间的距离
    """
    return np.linalg.norm(pt0 - pt1)

def order_points(pts):
    """
    按照左上、左下、右下、右上顺序排序坐标
    :params pts: 坐标点列表
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    (tl, bl) = leftMost[np.argsort(leftMost[:, 1]), :]
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, bl, br, tr], dtype="int32")

# 排序坐标点，左上、左下、右下、右上
approx = np.reshape(approx, (-1, 2))
approx = utils.order_points(approx)
approx = np.reshape(approx, (-1, 1, 2))

# 计算原矩形宽度和高度
rect_w = int(utils.calc_distance(approx[0][0], approx[-1][0]))
rect_h = int(utils.calc_distance(approx[0][0], approx[1][0]))

pts1 = np.float32(approx)
pts2 = np.float32([[0, 0], [0, rect_h], [rect_w, rect_h], [rect_w, 0]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst_img = cv2.warpPerspective(img_copied, M, (rect_w, rect_h))
```

经过上一个步骤的处理，已经可以用矩形将表格区域给框起来，在大部分情况下这个矩形都是有一点倾斜角度的，因此需要将这个矩形旋转正，并且将矩形中的表格区域给提取出来，这里我们用到了OpenCV的透视变换函数cv2.warpPerspective。

应用cv2.warpPerspective前需先使用cv2.getPerspectiveTransform得到转换矩阵。转换矩阵为3x3阶。为了便于理解上面的代码，先来看一个简单一点的例子：

```python
## 示例来源：https://zhuanlan.zhihu.com/p/37023649

# 定义对应的点
points1 = np.float32([[75,55], [340,55], [33,435], [400,433]])
points2 = np.float32([[0,0], [360,0], [0,420], [360,420]])

# points1为原始图片中的四边形；
# points2为希望变换后的四边形，即希望将四边形points1变换为比较正的矩形points2
# points1和points2中坐标点的位置应该是一一对应的，都会顺时针或者逆时针

# 计算得到转换矩阵
# M即为3*3的变换矩阵
M = cv2.getPerspectiveTransform(points1, points2)

# 实现透视变换转换
# 其中最后一个参数(360, 420)表示希望透视变换后图片的宽度和高度
img_processed = cv2.warpPerspective(img, M, (360, 420))
```

{% asset_img 7.jpg 透视变换示例 %}

有了上面这个示例，透视变换就会比较好理解。在示例中原图中的四个点是确定的，人工按照顺时针排列。而在成绩识别的代码中，这四个点是自动获取的，顺序不可控，所以需要先将这四个点按照左上、左下、右下、右上的顺序排列。变换后图片的宽度和高度可以通过矩形的两点之间的距离公式计算得到。通过透视变换后得到的表格图片如下图所示：

{% asset_img table_img.jpg 表格提取结果 %}

### 表格行提取

第二个阶段是根据表格中的边线，将表格切分为行。这个步骤主要使用OpenCV中膨胀和腐蚀操作来实现的。在进行表格切分前，同样需要对表格图片进行二值化预处理。

```python
def threshold(img, mode=cv2.THRESH_BINARY_INV, block_size=25, c=10):
    """
    将图片转换为二值化
    :param img: 图片
    :param mode: 转化为二值化的类型
    :param block_size:
    :param c:
    :return: 返回二值化后的OpenCV图像
    """
    img_gray = img.copy()
    # 判断图片是否为彩色的
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    # 自适应二值化
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, 
                                       cv2.ADAPTIVE_THRESH_MEAN_C, mode, block_size, c)
    return img_thresh

# 表格二值化
img_table_thresh = preprocess.threshold(img_table, block_size=25, c=30)
utils.save_image('output/img_table_thresh.jpg', img_table_thresh)    
```

{% asset_img img_table_thresh.jpg 表格二值化 %}

行的提取方法是分别对二值化后的表格图片进行纵向和横向的腐蚀和膨胀操作，使操作后的图片只显示竖线或者横线（如图所示），并将这2张图片进行按位与操作，即求得横线和竖线之间的交点，并获取交点坐标。

```python
def get_intersections(img_thresh, row_precision=90, col_precision=45):
    """
    取图片横线和竖线的交点
    :param img_thresh: 二值化图片
    :param row_precision: 横线提取精度
    :param col_precision: 竖线提取进度
    :return: 返回焦点坐标数组，分为x数组和y数组
    """
    row_lines = preprocess.recognize_rows(img_thresh, row_precision)
    col_lines = preprocess.recognize_cols(img_thresh, col_precision)
    intersections = cv2.bitwise_and(row_lines, col_lines)
    return np.where(intersections > 0)

def recognize_rows(img_thresh, row_precision=100):
    """
    从图片中识别横线
    :param img_thresh: 二值化图片
    :param row_precision: 行识别精度，数字越大识别出的横线越多。
    :return: 返回识别出横线后的OpenCV图片
    """
    img_copied = img_thresh.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_copied.shape[1]//row_precision, 1))
    img_eroded = cv2.erode(img_copied, kernel, iterations=1)
    img_dilate = cv2.dilate(img_eroded, kernel, iterations=1)
    return img_dilate

def recognize_cols(img_thresh, col_precision=45):
    """
    从图片中识别竖线
    :param img_thresh: 二值化图片
    :param col_precision: 列识别精度，数字越大识别出的竖线越多。
    :return: 返回识别出竖线后的OpenCV图片
    """
    img_copied = img_thresh.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(img_copied.shape[0]//col_precision)))
    img_eroded = cv2.erode(img_copied, kernel, iterations=1)
    img_dilate = cv2.dilate(img_eroded, kernel, iterations=1)
    return img_dilate

def filter_lines(lines, min_gap=50):
    """
    过滤距离太接近的线
    :param lines: 线条坐标
    :param min_gap: 线条之前的最小距离
    :return: 返回过滤后的线条坐标数组
    """
    line_list = []
    lines = np.sort(lines)
    for i in range(len(lines) - 1):
        if lines[i + 1] - lines[i] > min_gap:
            line_list.append(lines[i])
    line_list.append(lines[len(lines) - 1])
    return line_list

# 获取表格的焦点坐标
xs, ys = get_intersections(img_table_thresh, 60)
x_lines = filter_lines(xs)
```


下图为实际运行结果，从左到右依次为，行提取、列提取、行列交点

{% asset_img rows.png 横线识别 %}
{% asset_img cols.png 竖线识别 %}
{% asset_img intersections.jpg 交点识别 %}

下图为使用`x_lines`中的坐标在表格图片中描出行后的图片。

{% asset_img img_table_lined.jpg 行识别 %}

从图片中可以看到识别出来的行还存在几个问题：

1. 由于纸张存在弯曲以及行线本身不直，导致提取的横线和表格行之间的边线并不能完全重合；
2. 提取的横线部分会切到序号和成绩的数字；
3. 表格的前三行（项目、日期、表头）其实是不需要的；

因此在后续行的处理过程中，需要解决上述的几个问题，使序号列和成绩列提取更加准确。

> 在这里补充一下之前进行的尝试，在提取行的时候，由于行和列的坐标点已经都获取到了（代码中的xs和ys），其实可以直接在表格图片中将表格中的单元格都一次性切出来，如下图：
>
> {% asset_img img_table_lined2.jpg 行列边界和交点 %}
>
> 但是这样处理会存在一些问题，比如行无法进行角度校正，所以会导致部分数字被切掉一点（第一行的87分的成绩）；表格上面三行的列没有与成绩部分的列对齐，所以在包含上面三行的情况下，下半部分的有些列被切多了，如果只提取序号和成绩那么并不影响，但是如果需要提取所属团队或者曲目的时候就会发现这2列被一切为2了。
>
> 当然这也是有解决办法的，由于表格是赛事组委会自己设计的，可以考虑把表头区域和评分区域之间的分割线加粗，那么可以通过腐蚀操作提取两个区域的分割线。

### 表格单元格提取

```python
def cut_image_rows(img_rotated, x_lines, margin=10):
    """
    从原图中切出行
    :param img_rotated: 校正后的图片
    :param x_lines: x坐标
    :param margin: 切图的边距
    :return: 每行返回一次，返回OpenCV图片和行坐标
    """
    for xi in range(len(x_lines)-1):
        x0, x1 = x_lines[xi], x_lines[xi+1]        
        img_row = img_rotated[x0-margin:x1+margin, :]
        yield img_row, xi


for img_row, img_i in cut_image_rows(img_table, x_lines, 10):
    # 去除表头和表格其他部分
    if img_row.shape[0] > 180:
        continue
```

这里使用之前提取的行坐标，将表格中的每一行单独切分为小图片，由于表格的前三行比其他行宽，这里使用宽度把前三行过滤掉。下图看上去还是比较规整的，角度也基本上是水平的。

{% asset_img img_row.jpg 行提取 %}

而下面这张图并不是那么规整，行线在右侧有点上翘，数字87由于太贴近上边缘，使数字7被切掉了一点。

{% asset_img img_row_3.jpg 行提取 %}

为了保证切取得每一行都是尽量水平的，并且能保留尽量完整的数字，需要对每一张行图片进行角度校正。校正的方法是先用之前的recognize_rows方法提取行图片中的横线，由于切图的时候会在上下各增加10个像素的边界，所以一般切图后的图片中都会带有表格边线，至少会有一条边线。然后使用霍夫变换方法HoughLinesP识别横线，提取这条横线的坐标后既可以用角度公式计算这条直线的倾斜角度，根据这个角度旋转图片即可。

```python
def rotate(img, angle):
    """
    旋转图片
    :param img: 图片
    :param angle: 旋转角度
    :return: 返回旋转后的OpenCV图片
    """
    h, w, _ = img.shape
    center = (w // 2, h // 2)
    mtx = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_copied = img.copy()
    img_rotated = cv2.warpAffine(img_copied, mtx, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    ## BORDER_WRAP
    ## 图像边界的处理方式: BORDER_WRAP
    ## cdefgh|abcdefgh|abcdef
    return img_rotated

def angle_correct(img, row_precision=150, min_line_length=200, max_line_gap=15):
    """
    图片角度校正
    :param img: 图片
    :param row_precision: 行识别精度，数字越大识别出的横线越多。
    :param min_line_length:
    :param max_line_gap:
    :return: 返回角度已经校正后的OpenCV图片
    """
    img_copied = img.copy()
    # 图片二值化
    img_thresh = threshold(img_copied)
    # 从二值化图片中识别横线
    img_lined = recognize_rows(img_thresh, row_precision)
    # 霍夫变换，提取横线坐标
    lines = cv2.HoughLinesP(img_lined, 1, np.pi / 180, 80, min_line_length, max_line_gap)
    # 将三维数组转换为(-1, 4)的二维数组
    lines = np.reshape(lines, (len(lines), 4))
    # 找到最长的直线
    angle = utils.get_correct_angle(lines, img_copied.shape[1]/3)
    # 旋转图片
    img_rotated = img_copied
    if angle != 0.0:
        img_rotated = utils.rotate(img_copied, angle)
    return img_rotated, angle


# 校正行图片
img_row_rotated, angle = preprocess.angle_correct(img_row)
```

下图为校正后的行图片，可以看到经过校正后图片还是比较平整的，原来被切掉的数字7也比之前有所改观。

{% asset_img img_row_3_rotated.jpg 行角度矫正 %}

将行图片当成只有一行的表格，使用之前识别表格的方式提取行列的交叉点，区别是这次使用列坐标，识别行中的列。

```python
# 行图片二值化
img_row_thresh = preprocess.threshold(img_row_rotated)

# 为每行补充上下两条横线，用于后续交叉点的识别
h, w = img_row_thresh.shape
cv2.line(img_row_thresh, (0, 2), (w, 2), (255), 5)
cv2.line(img_row_thresh, (0, h-6), (w, h-6), (255), 5)

# 获取交点坐标
xs, ys = get_intersections(img_row_thresh, 1, 1)
x_lines = filter_lines(xs)
y_lines = filter_lines(ys) 
```

这里作了一个小的处理，由于在切行图片的时候不能保证一定会有上下两条行边线，导致无法准确提取行列交点，所以我手工在行图片的上下两端分别绘制两条行的边线，来替代原始的行边线，这样可以保证准确的获取行和列的交点坐标。分别将行、列以及交点绘制到图片上，可以看到如下输出结果：

{% asset_img img_row_3_inter.jpg 单元格识别 %}

列的坐标点准确识别后，提取每一列还是比较容易的。由于需求上我们只需要第一列（序号列）和最后一列（成绩列），因此我们只需要分别获取前2个坐标和后2个坐标即可。

```python
# 切割列
## 提取第一列ID列
img_cell_id = img_row_rotated[:, y_lines[0]:y_lines[1]]

## 提取最后一列成绩列
img_cell_score = img_row_rotated[:, y_lines[-2]:y_lines[-1]]
```

{% asset_img img_cells.jpg 序号列和成绩列提取 %}

到这里我们已经成功提取了图片中序号列和成绩列。由于序号是印刷体，而成绩属于手写体，需要采用不同的方法来识别这2类数字。针对印刷体，我们采用Tesseract-OCR来识别；手写体的数字采用深度学习的方法来提取。

### 印刷体数字的识别



### 手写数字提取与切分



### 手写数字模型的构建和训练



### 成绩识别



### 成绩识别服务的构建



### 总结