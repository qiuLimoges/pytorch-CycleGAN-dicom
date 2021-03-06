## 神经网络CycleGAN对医学图像格式文件DICOM的操作

2019年最新的研究文章显示了使用cycleGAN对dicom文件进行操作的可能性。

在实际应用中，研究显示了对磁共振图像映射成CT图像以及将CBCT图像映射成常规CT的可能性。我觉得很有实际用途，于是在CycleGAN原作者的程序代码中，添加了可以进行dicom图像操作的辅助代码。

目前已经可以在机器上进行CT和磁共振的数据训练。不过还没有进行检验。我把源代码先传上来，慢慢的修改。

2020年1月24日：主要是应用Pydicom组件进行Dicom图像的读取，然后以Numpy Array的形式赋给PIL.Image对象，注意要将图像转换成“I”格式也就是32位图像。由于源代码主要是处理彩色RGB8位图像，而Dicom图像是16位的灰度图，所以在改编时遇到了不少的困难。最后找到的技术路径是将16位的dicom图转换为32位的PIL.Image对象输入程序。程序内部在输入神经网络前实际上将其转换成0到1的浮点数。在输出端将其转换成16位的整型二维数列（实际是RGB的三维数列，不过只需读取0，X，Y两维），封装成NumpyArray就可以重新写入Dicom文件。代码目前还没有对Dicom文件进行全面写入，主要是标记部分没有转入。

2020年1月25日：完成Gighub的同步操作。总算可以一次修改代码，到处备份。不过应用ct2IRM产生的处理文件，仍旧不能有效的生成可读的DIcom文件。怀疑主要问题还是Dicom数据转换，因为在经过神经网络处理后出来的文件，像素最大值接近6万。很显然是转换数据时出了问题。另外，不知道是不是需要将DIcom原始数据转换成HU值然后在输入网络。感觉似乎是不需要这样做的。

2020年1月26日：数据转入神经网络的关键是在./datat/base_dataset.py中的BaseDataset类。在Get_transform的函数中使用了Torchvision.transform函数进行数据的预处理和归一化Normalisation。需要找到一个可靠的方法在神经网络的出口处将数据转换成dicom图像需要的格式。坏消息是Transforms.ToTensor好像只支持将八位数据转换成0-1的浮点数，对于其他类型，返还元数据不予处理。看来则需要我在转入之前就要先将数据手工转换好。否则不会成功。具体详情可以参考 torchvision.transforms.totensor的说明。

2020.01.27 Effet SimpleITK semble bien pour decoder dicom image. existe un code example pour ecrire sous forme Dicom: https://itk.org/SimpleITKDoxygen/html/DicomSeriesReadModifyWrite_2DicomSeriesReadModifySeriesWrite_8py-example.html

2020.01.30: # Deep Learning: A Review for the Radiation Oncologist:https://www.frontiersin.org/articles/10.3389/fonc.2019.00977/full
propablement je vais changer le direction vers le contourage automatique puisqu'il s'agit d'un seule CNN, plus vite et mature dans le domain medicale. 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ0NjEzMTU0MCwxNjgxMDcxOTc3LDk2Nz
A4NTA5OCwxMzk3OTAzNjIzXX0=
-->