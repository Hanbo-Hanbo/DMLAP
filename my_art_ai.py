import cv2  # 引入摄像头工具包（小名叫 cv2）
import mediapipe as mp  # 引入人工智能专家包（小名叫 mp）

# --- 第一步：叫专家进场 ---
mp_hands = mp.solutions.hands  # 从专家包里找到“专门看手”的部门
# 雇佣一名“看手专家”，并告诉他：
# min_detection_confidence: 只有你【第 1 次】找手时，超过 0.7 的把握才算数
# min_tracking_confidence: 一旦找到了，后面【追踪】手的时候，超过 0.5 的把握就跟着走
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) 

mp_draw = mp.solutions.drawing_utils  # 准备一支“画笔”，用来在画面上连线

# --- 第二步：打开摄像头 ---
cap = cv2.VideoCapture(0)  # 这里的 0 代表电脑自带的第 1 个摄像头

# 只要摄像头是开着的，就一直循环运行
while cap.isOpened():
    # success: 拍照是否成功（True/False）
    # image: 拍到的那张照片
    success, image = cap.read()
    
    if not success:  # 如果拍照失败（比如摄像头断了）
        print("没拍到照片，跳过这次")
        continue

    # --- 第三步：给图片做“变身” ---
    # 摄像头拍出来的颜色顺序是 BGR（蓝绿红），但 AI 喜欢 RGB（红绿蓝）
    # 我们要给照片换个色，AI 才能看懂
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 第四步：AI 开始扫描（卷积运算就在这里面偷偷发生） ---
    # 让 AI 专家处理这张图片，结果存进 results 里
    results = hands.process(image_rgb)

    # --- 第五步：处理扫描出来的结果 ---
    # 如果 results 里的 multi_hand_landmarks（找到的手部关节）不是空的
    if results.multi_hand_landmarks:
        # 因为画面里可能有好几只手，我们要一只一只处理
        for hand_landmarks in results.multi_hand_landmarks:
            
            # 1. 自动画出骨架：在 image 上，把找到的关节 hand_landmarks 连上线
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 2. 获取具体坐标：这里有 21 个点的 x, y 坐标
            # 我们拿食指尖（点 8）和中指尖（点 12）来举例
            # 这里的坐标是比例（0 到 1），我们要用它乘以图片的宽和高
            h, w, c = image.shape # 拿到图片的【高、宽、颜色通道】
            
            # 拿到食指尖的具体位置
            index_finger_tip = hand_landmarks.landmark[8]
            # 拿到中指尖的具体位置
            middle_finger_tip = hand_landmarks.landmark[12]
            
            # --- 简单的“剪刀石头布”判断逻辑 ---
            # 如果食指和中指都比手掌根部（点 0）高，我们暂且认为这是“剪刀”
            if index_finger_tip.y < hand_landmarks.landmark[6].y and \
               middle_finger_tip.y < hand_landmarks.landmark[10].y:
                cv2.putText(image, "SCISSORS!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 第六步：把结果显示在屏幕上 ---
    cv2.imshow('My AI Interaction', image)

    # 如果你按下键盘上的 'q' 键（quit），就退出程序
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 第七步：善后工作 ---
cap.release()  # 关掉摄像头
cv2.destroyAllWindows()  # 关掉所有弹出的窗口