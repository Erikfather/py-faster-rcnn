# py-faster-rcnn
## 利用py-faster-rcnn做的第一个实验室项目代码记录
### 各文件功能简介
  （1）ceshi.py文件是为修改图片名做的测试，目的是识别图片中的目标，将识别到的目标数量加入图片名,如原图片名为：2018-05-03-15-33.jpg,现在识别到这张图片里有两个person,一个car,则图片名被修改为：2018-05-03-15-33-02-01.jpg;<br>这部分功能在changename.py和detectv8.py具体实现了;<br>
  （2）detectv1.py--detectv2.py是建立了文件系统后，在识别检测目标的基础上，为了统计每个图片中目标个数，并写入num.txt文件所做的测试<br>
  （3）detectv3.py--detectv5.py是合成（画变化部分包围框记录--这部分是学长做的）和（读写num.txt）所做的优化测试过程;<br>
  （4）detectv6.py是为了建立缩略图系统同时删除了num.txt功能文件所做的测试；<br>
  （5）detectv7.py是学长为了测试在读取完原始图片后能否将其删除所做的测试，为下面利用爬虫做准备,这个文件注释了缩略图部分；<br>
  （6）new.py和work.py是一开始建立文件管理系统存取识别目标图片所做的测试;<br>
  （7）savepics2.py是爬取网页图片存入indata文件夹的过程;<br>
  
## 目前的功能是：
#### 先运行savepics2.py文件将网页上图片保存到0503indata文件夹，然后运行detectv8.py将0503indata里面的图片全部重命名放到0503data里面,格式如（1）所述，同时自动将0503indata文件夹下所有图片删除，等待下次爬取图片存入；然后开始读取0503data里面的文件，并将检测结果保存到0503pic_results文件夹下面，全部处理完后自动删除0503data里面所有图片，等待接收下次处理过的爬取图片，进行下一次工作。
