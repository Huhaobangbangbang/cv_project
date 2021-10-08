import numpy as np
import cv2
import math
import random
def non_max_suppression_fast(boxes, overlapThresh):
    """将矩形框中的矩形框去掉"""
    # 空数组检测
    if len(boxes) == 0:
        return []
    # 将类型转为float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    # 四个坐标数组
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算面积数组
    idxs = np.argsort(y2)  # 返回的是右下角坐标从小到大的索引值

    # 开始遍历删除重复的框
    while len(idxs) > 0:
        # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # 找到剩下的其余框中最大的坐标x1y1，和最小的坐标x2y2,
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算重叠面积占对应框的比例
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        # 如果占比大于阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


"""DBSCAN算法的核心是“延伸”。先找到一个未访问的点p，若该点是核心点，则创建一个新的簇C，
将其邻域中的点放入该簇，并遍历其邻域中的点，若其邻域中有点q为核心点，则将q的邻域内的点也划入簇C，
直到C不再扩展。直到最后所有的点都标记为已访问。"""

"""
DBSCAN算法实现
1、标记点是否被访问：我设置了两个列表，一个存放未访问的点unvisited，一个存放已访问的点visited。每次访问一个点，unvisited列表remove该点，visited列表append该点，以此来实现点的标记改变。
2、C作为输出结果，初始时是一个长度为所有点的个数的值全为-1的列表。之后修改点对应的索引的值来设置点属于哪个簇"""

def dist(t1, t2):
    # 计算两个点之间的欧式距离，参数为两个元组
    dis = math.sqrt((np.power((t1[0]-t2[0]),2) + np.power((t1[1]-t2[1]),2)))
    return dis

def dbscan(spot_pos_list, Eps):
    # DBSCAN算法，spot_pos_list为矩形框的中心，Eps为指定半径参数(这里设置为图书馆章的大小)
    # we want to return the new_pick which contain x,y,w,h
    # we are getting the new spot_pos_list sorted by the distance of spot
    new_pick = []
    for (startX, startY, endX, endY) in spot_pos_list:
        if len(new_pick) == 0:
            new_pick.append([startX, startY, endX, endY])
        else:
            flag = 0
            print(new_pick)
            for index in range(len(new_pick)):
                # 表示属于一个聚类,就更新这个聚类的x,y,w,h
                # 如果两个轮廓重叠，则被认定一个聚类
                # if abs(startY - new_pick[index][1]) < new_pick[index][3]+endY and abs(startX-new_pick[index][0]) < new_pick[index][2]+endX:
                #     x = (startX+new_pick[index][0])/2
                #     y = (startY+new_pick[index][1])/2
                #     w = (endX+new_pick[index][2]+abs(startY - new_pick[index][1]))
                #     h = (endY+new_pick[index][3]+abs(startX-new_pick[index][0]))
                #     new_pick[0] = [x, y, w, h]
                #     print('hello')
                #     break
                # 如果两个轮廓的距离小于阈值则被认为是同一个聚类
                x1 = (startX + endX)/2
                y1 = (startY + endY)/2
                x2 = (new_pick[index][0] + new_pick[index][2])/2
                y2 = (new_pick[index][1] + new_pick[index][3])/2
                dis = dist([x1, y1],[x2, y2])
                if dis <= Eps:
                    x = min(startX, new_pick[index][0])
                    y = min(startY, new_pick[index][1])
                    w = max(endX, new_pick[index][2])
                    h = max(endY, new_pick[index][3])
                    new_pick[index] = [x, y, w, h]
                    print('shit')
                    flag = 1
                # 表示不是一个聚类
                else:
                    pass
            if flag == 0:
               new_pick.append([startX, startY, endX, endY])

    return new_pick

def get_word_area(img_path):
    """得到检测图像中的文本区域，画出轮廓"""
    mser = cv2.MSER_create()
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    '''NMS是经常伴随图像区域检测的算法，作用是去除重复的区域，
    在人脸识别、物体检测等领域都经常使用，全称是非极大值抑制（non maximum suppression），
    就是抑制不是极大值的元素，所以用在这里就是抑制不是最大框的框，也就是去除大框中包含的小框'''
    # 绘制目前的矩形文本框
    mser = cv2.MSER_create()
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        keep.append([x, y, x + w, y + h])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
    # showing the pic before the remove the small box
    # cv2.namedWindow("before NMS", 0)
    # cv2.resizeWindow("before NMS", 800, 640)
    # cv2.imshow("After NMS", vis)
    print("[x] %d initial bounding boxes" % (len(keep)))
    # 筛选不重复的矩形框
    # we are getting the information of rectangle which contain the  x,y,w,h
    keep2 = np.array(keep)
    pick = non_max_suppression_fast(keep2, 0.5)
    # 将相邻的矩形框聚类
    # 将相邻的矩形框聚类,we are getting the new_pick which contain (startX, startY, endX, endY)
    print("after applying non-maximum, %d bounding boxes" % (len(pick)))
    orig = img.copy()
    new_pick = dbscan(pick, 500)
    for (startX, startY, endX, endY) in new_pick:
        print(startX, startY, endX, endY)
        cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 185, 120), 2)

    cv2.namedWindow("After cluster", 0)
    cv2.resizeWindow("After clustering", 800, 640)
    cv2.imshow("After clustering", orig)
    cv2.imwrite('afterclustering.jpg', orig)

    if cv2.waitKey(0) == 9:
        cv2.destroyAllWindows()
    return vis


if __name__ == '__main__':
    img_path = 'sample3.png'
    vis = get_word_area(img_path)
