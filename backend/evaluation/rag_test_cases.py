"""
RAG 评估测试数据集 —— 严格基于 fTaoBao 电商平台服务指南中的真实内容

每个测试用例包含:
  - question: 用户问题
  - expected_keywords: 期望回答中出现的关键词（用于检索质量验证）
  - forbidden_keywords: 不应该出现的关键词（用于幻觉检测）
  - expected_answer: 期望的回答（用于生成质量/忠诚度验证的参考）
  - expected_context_topics: 检索到的上下文应包含的主题

说明:
  - 所有 30 个测试用例的答案都可以在 fTaoBao_knowledge_base.md 中找到明确依据
  - 难度分布：easy 15 / medium 10 / hard 5
  - 覆盖场景：平台基础信息、客服联系、8 大商品分类、具体商品规格
"""

from typing import List, Dict, Optional


def get_default_test_cases() -> List[Dict]:
    """返回内置的评估测试用例（30 个，全部基于知识库真实内容）"""
    return [
        # ===== 平台基础信息（easy）=====
        {
            "id": "platform_service_001",
            "category": "服务承诺",
            "question": "fTaoBao 有哪些服务承诺？",
            "expected_keywords": ["正品保障", "七天无理由", "极速物流", "专业客服", "价保服务", "闪赔"],
            "forbidden_keywords": ["免费赠送", "7 天无理由退一赔三"],
            "expected_answer": "fTaoBao 提供六大服务承诺：1. 正品保障，假一赔十；2. 七天无理由退换货；3. 极速物流，主要城市 24 小时送达；4. 专业客服，7×24 小时在线，平均响应时间 30 秒；5. 价保服务，下单后 7 天内降价全额退还差价；6. 闪赔服务，质量问题确认后 1 小时内完成退款。",
            "expected_context_topics": ["服务承诺", "正品保障", "七天无理由", "极速物流", "价保服务", "闪赔"],
            "difficulty": "easy",
        },
        {
            "id": "platform_company_001",
            "category": "平台信息",
            "question": "fTaoBao 公司的注册地址在哪里？",
            "expected_keywords": ["北京市朝阳区", "建国路 88 号", "fTaoBao 大厦"],
            "forbidden_keywords": ["上海市", "深圳", "保密"],
            "expected_answer": "fTaoBao 科技发展有限公司注册地址为北京市朝阳区建国路 88 号 fTaoBao 大厦。客服中心地址位于上海市浦东新区张江高科技园区博云路 2 号。",
            "expected_context_topics": ["公司全称", "注册地址", "客服中心地址"],
            "difficulty": "easy",
        },
        {
            "id": "platform_business_hours_001",
            "category": "营业时间",
            "question": "人工客服的工作时间是几点到几点？",
            "expected_keywords": ["8:00", "22:00", "每日", "人工客服", "智能机器人", "7×24"],
            "forbidden_keywords": ["24 小时人工", "朝九晚五"],
            "expected_answer": "人工客服工作时间为每日 8:00-22:00。非人工时段，智能机器人客服提供 7×24 小时不间断服务，覆盖 2000+ 常见问题。",
            "expected_context_topics": ["人工客服", "营业时间", "智能机器人"],
            "difficulty": "easy",
        },
        {
            "id": "platform_hours_002",
            "category": "营业时间",
            "question": "节假日能正常下单吗？客服节假日营业吗？",
            "expected_keywords": ["节假日", "正常营业", "7×24", "客服", "物流"],
            "forbidden_keywords": ["节假日放假", "节假日不能下单"],
            "expected_answer": "fTaoBao 平台网站及 APP 7×24 小时不间断营业，法定节假日正常营业，客服及物流正常运营。",
            "expected_context_topics": ["平台营业时间", "节假日服务"],
            "difficulty": "easy",
        },

        # ===== 客服联系方式（easy/medium）=====
        {
            "id": "contact_phone_001",
            "category": "客服联系",
            "question": "官方客服电话是多少？",
            "expected_keywords": ["400-888-8888", "7×24", "客服热线"],
            "forbidden_keywords": ["110", "私人手机号", "400-888-8889"],
            "expected_answer": "fTaoBao 官方客服热线为 400-888-8888，提供 7×24 小时全天候服务。此外还设有订单查询专线 400-888-8889、售后服务专线 400-888-8890、投诉建议热线 400-888-8891、VIP 会员专线 400-888-8899。",
            "expected_context_topics": ["客服热线", "400-888-8888"],
            "difficulty": "easy",
        },
        {
            "id": "contact_phone_002",
            "category": "客服联系",
            "question": "VIP 会员专线电话是什么？",
            "expected_keywords": ["400-888-8899", "7×24", "VIP", "专属服务"],
            "forbidden_keywords": ["400-888-8888"],
            "expected_answer": "VIP 会员专线为 400-888-8899，提供 7×24 小时会员专属服务。",
            "expected_context_topics": ["VIP 会员专线", "400-888-8899"],
            "difficulty": "easy",
        },
        {
            "id": "contact_email_001",
            "category": "客服联系",
            "question": "售后问题应该发邮件到哪个邮箱？",
            "expected_keywords": ["support@ftaoabao.com", "24 小时", "售后", "邮箱"],
            "forbidden_keywords": ["service@ftaoabao.com", "order@ftaoabao.com"],
            "expected_answer": "售后服务邮箱为 support@ftaoabao.com，通常 24 小时内回复。客户服务邮箱为 service@ftaoabao.com，订单咨询邮箱为 order@ftaoabao.com。",
            "expected_context_topics": ["售后邮箱", "support@ftaoabao.com"],
            "difficulty": "easy",
        },

        # ===== 商品分类（easy/medium）=====
        {
            "id": "category_001",
            "category": "商品分类",
            "question": "你们平台有多少个商品分类？分别是什么？",
            "expected_keywords": ["手机数码", "电脑办公", "家用电器", "服饰鞋包", "食品生鲜", "美妆护肤", "运动户外", "家居生活", "八"],
            "forbidden_keywords": ["军火", "处方药", "十个分类"],
            "expected_answer": "fTaoBao 平台目前提供八大核心商品分类：手机数码、电脑办公、家用电器、服饰鞋包、食品生鲜、美妆护肤、运动户外、家居生活，覆盖消费者日常生活的各个方面。",
            "expected_context_topics": ["商品分类", "八大核心商品分类"],
            "difficulty": "easy",
        },
        {
            "id": "category_002",
            "category": "商品分类",
            "question": "商品编号是怎么命名的？有什么规则？",
            "expected_keywords": ["CAT", "分类缩写", "序号", "商品编号"],
            "forbidden_keywords": ["随机编号", "二维码"],
            "expected_answer": "每款商品在平台中都有唯一的商品编号，格式为 CAT-[分类缩写]-[序号]。例如手机数码类商品前缀为 CAT-PHONE-001 至 CAT-PHONE-025，电脑办公类为 CAT-PC-001 至 CAT-PC-025，家用电器类为 CAT-HOME-001 至 CAT-HOME-025，服饰鞋包类为 CAT-CLOTH-001 至 CAT-CLOTH-025，食品生鲜类为 CAT-FOOD-001 至 CAT-FOOD-025，美妆护肤类为 CAT-BEAUTY-001 至 CAT-BEAUTY-025，运动户外类为 CAT-SPORT-001 至 CAT-SPORT-025，家居生活类为 CAT-LIFE-001 至 CAT-LIFE-025。",
            "expected_context_topics": ["商品编号查询说明", "CAT-[分类缩写]-[序号]"],
            "difficulty": "hard",
        },

        # ===== 手机数码商品（easy/medium/hard）=====
        {
            "id": "phone_iphone_001",
            "category": "手机数码",
            "question": "iPhone 15 Pro Max 的售价是多少？",
            "expected_keywords": ["8999", "iPhone 15 Pro Max", "Apple"],
            "forbidden_keywords": ["9999", "7999"],
            "expected_answer": "iPhone 15 Pro Max（商品编号 CAT-PHONE-001）售价为 ¥8999 元，配有 6.7 英寸 ProMotion 自适应刷新率显示屏，A17 Pro 芯片，4800 万像素主摄，5 倍光学变焦，支持 USB-C 接口，钛金属边框，IP68 防水防尘。",
            "expected_context_topics": ["iPhone 15 Pro Max", "CAT-PHONE-001", "售价 8999 元"],
            "difficulty": "easy",
        },
        {
            "id": "phone_huawei_001",
            "category": "手机数码",
            "question": "华为 Mate 60 Pro 的价格和芯片是什么？",
            "expected_keywords": ["6999", "麒麟 9000S", "卫星通话", "华为", "Mate 60 Pro"],
            "forbidden_keywords": ["骁龙芯片", "7999"],
            "expected_answer": "华为 Mate 60 Pro（商品编号 CAT-PHONE-004）售价为 ¥6999 元，搭载自研麒麟 9000S 芯片，是全球首款支持卫星通话的大众智能手机。配备 6.82 英寸 LTPO OLED 曲面屏，昆仑玻璃二代，12GB+512GB 存储，5000mAh 大电池，88W 有线快充，50W 无线充电。",
            "expected_context_topics": ["华为 Mate 60 Pro", "CAT-PHONE-004", "麒麟 9000S", "卫星通话"],
            "difficulty": "easy",
        },
        {
            "id": "phone_xiaomi_001",
            "category": "手机数码",
            "question": "小米 14 Ultra 用的是什么处理器？价格多少？",
            "expected_keywords": ["骁龙 8 Gen3", "5999", "徕卡", "小米 14 Ultra"],
            "forbidden_keywords": ["联发科", "6999"],
            "expected_answer": "小米 14 Ultra（商品编号 CAT-PHONE-008）售价为 ¥5999 元，搭载骁龙 8 Gen3 处理器，配有徕卡可变光圈主摄（f/1.63-f/4.0），75mm 浮动长焦，6.73 英寸 2K LTPO OLED 全等深微曲屏，5300mAh 电池，90W 有线快充。",
            "expected_context_topics": ["小米 14 Ultra", "CAT-PHONE-008", "骁龙 8 Gen3", "徕卡光学"],
            "difficulty": "easy",
        },
        {
            "id": "phone_oppo_001",
            "category": "手机数码",
            "question": "OPPO Find X7 Ultra 有什么特色？售价多少？",
            "expected_keywords": ["5999", "双潜望四摄", "哈苏影像", "OPPO Find X7 Ultra", "卫星通信"],
            "forbidden_keywords": ["天玑芯片", "6999"],
            "expected_answer": "OPPO Find X7 Ultra（商品编号 CAT-PHONE-012）售价为 ¥5999 元，搭载骁龙 8 Gen3 处理器，配备双潜望四摄哈苏影像系统（5000 万一英寸主摄+5000 万超广角+5000 万 2.8 倍长焦+6400 万 5 倍潜望长焦），支持卫星通信功能，配备 5000mAh 电池和 100W 超级闪充。",
            "expected_context_topics": ["OPPO Find X7 Ultra", "CAT-PHONE-012", "哈苏影像", "双潜望四摄"],
            "difficulty": "hard",
        },
        {
            "id": "phone_samsung_001",
            "category": "手机数码",
            "question": "三星 Galaxy S24 Ultra 的价格是多少？有什么特别的功能？",
            "expected_keywords": ["9699", "S Pen", "骁龙 8 Gen3 for Galaxy", "AI 智能"],
            "forbidden_keywords": ["7999", "手写笔不支持"],
            "expected_answer": "三星 Galaxy S24 Ultra（商品编号 CAT-PHONE-018）售价为 ¥9699 元，搭载骁龙 8 Gen3 for Galaxy 处理器，内置 S Pen 手写笔，支持 AI 智能助手，具备 2 亿像素主摄+1200 万超广角+1000 万 3 倍长焦+1000 万 5 倍长焦，5000mAh 电池，45W 超级快充，钛金属边框。",
            "expected_context_topics": ["三星 Galaxy S24 Ultra", "CAT-PHONE-018", "S Pen", "AI 智能"],
            "difficulty": "easy",
        },
        {
            "id": "phone_honor_001",
            "category": "手机数码",
            "question": "荣耀 Magic6 Pro 的青海湖电池有什么特点？价格是多少？",
            "expected_keywords": ["5699", "青海湖电池", "鸿燕通信", "北斗卫星", "荣耀 Magic6 Pro"],
            "forbidden_keywords": ["石墨烯电池", "太阳能充电"],
            "expected_answer": "荣耀 Magic6 Pro（商品编号 CAT-PHONE-020）售价为 ¥5699 元，配备青海湖电池技术，续航持久，搭载荣耀鸿燕通信，支持双向北斗卫星消息。具体规格为：6.8 英寸 OLED 巨犀玻璃屏，骁龙 8 Gen3 处理器，5000 万像素主摄+5000 万超广角+1.8 亿像素 5 倍潜望长焦，80W 有线+80W 无线双超级快充。",
            "expected_context_topics": ["荣耀 Magic6 Pro", "CAT-PHONE-020", "青海湖电池", "鸿燕通信"],
            "difficulty": "medium",
        },

        # ===== 电脑办公产品（medium）=====
        {
            "id": "pc_macbook_001",
            "category": "电脑办公",
            "question": "MacBook Pro 16 寸（M3 Max）的售价和配置是什么？",
            "expected_keywords": ["19999", "M3 Max", "32GB", "1TB", "MacBook Pro 16"],
            "forbidden_keywords": ["14999", "M3 Pro", "16GB"],
            "expected_answer": "MacBook Pro 16 寸（商品编号 CAT-PC-001）售价为 ¥19999 元，配备 16.2 英寸 Liquid Retina XDR 显示屏（3456×2234），Apple M3 Max 芯片（16 核 CPU、40 核 GPU），32GB 统一内存，1TB SSD 存储。配有 MagSafe 3 充电口、三个雷雳 4 接口、HDMI 接口、SDXC 卡槽，支持 Wi-Fi 6E、蓝牙 5.3。",
            "expected_context_topics": ["MacBook Pro 16寸", "CAT-PC-001", "M3 Max 芯片"],
            "difficulty": "hard",
        },
        {
            "id": "pc_lenovo_001",
            "category": "电脑办公",
            "question": "联想 ThinkPad X1 Carbon 的价格和特点是什么？",
            "expected_keywords": ["8999", "14 寸", "军标测试", "指纹识别", "ThinkPad X1 Carbon"],
            "forbidden_keywords": ["游戏本", "5999"],
            "expected_answer": "联想 ThinkPad X1 Carbon（商品编号 CAT-PC-005）售价为 ¥8999 元，配备 14 英寸 WUXGA IPS 防眩光屏（1920×1200），第 13 代英特尔酷睿 i7-1365U 处理器，32GB DDR5 内存，1TB NVMe SSD，重量仅 1.12kg。通过 12 项军标测试，耐用可靠，配备 TrackPoint 指点杆+全尺寸背光键盘，支持指纹识别、人脸登录。",
            "expected_context_topics": ["联想 ThinkPad X1 Carbon", "CAT-PC-005", "商务旗舰", "军标测试"],
            "difficulty": "medium",
        },
        {
            "id": "pc_ipad_001",
            "category": "电脑办公",
            "question": "iPad Pro M4 2024 的价格和屏幕参数？",
            "expected_keywords": ["7999", "13 寸", "Liquid Retina XDR", "Mini LED", "iPad Pro M4"],
            "forbidden_keywords": ["10.9 寸", "4799"],
            "expected_answer": "iPad Pro M4 2024（商品编号 CAT-PC-008）售价为 ¥7999 元，配备 13 英寸 Liquid Retina XDR 显示屏（Mini LED，1600 尼特），Apple M4 芯片（9 核 CPU、10 核 GPU），8GB 内存，256GB 存储。支持妙控键盘，配备 LiDAR 扫描仪，支持 Wi-Fi 6E、蓝牙 5.3，四扬声器音响系统。",
            "expected_context_topics": ["iPad Pro M4 2024", "CAT-PC-008", "Liquid Retina XDR"],
            "difficulty": "easy",
        },

        # ===== 家用电器产品（easy/medium/hard）=====
        {
            "id": "home_dyson_v15_001",
            "category": "家用电器",
            "question": "戴森 V15 无线吸尘器多少钱？有什么核心技术？",
            "expected_keywords": ["4999", "激光探测", "Hyperdymium", "60 分钟", "戴森 V15"],
            "forbidden_keywords": ["2999", "普通电机"],
            "expected_answer": "戴森 V15 无线吸尘器（商品编号 CAT-HOME-001）售价为 ¥4999 元，核心技术包括：激光探测灰尘技术，可精准识别微尘；Hyperdymium 马达，吸力强劲；60 分钟长效续航（节能模式）；高级过滤系统，捕获 99.99% 小至 0.3 微米的微尘。LCD 屏幕实时显示灰尘数量和大小。",
            "expected_context_topics": ["戴森 V15 无线吸尘器", "CAT-HOME-001", "激光探测"],
            "difficulty": "hard",
        },
        {
            "id": "home_gree_ac_001",
            "category": "家用电器",
            "question": "格力 1.5 匹空调的价格和保修政策？",
            "expected_keywords": ["3299", "新一级能效", "自清洁", "整机 6 年", "压缩机 10 年", "格力空调"],
            "forbidden_keywords": ["一年保修", "2999"],
            "expected_answer": "格力 1.5 匹云佳空调（商品编号 CAT-HOME-003）售价为 ¥3299 元，为新一级能效，省电节能，配备变频压缩机，温度精准控制，具有 56℃ 高温自清洁功能。格力凌达压缩机品质保证，独立除湿功能，四种睡眠模式，7 档风速调节，静音低至 18dB。整机 6 年免费保修，压缩机 10 年包修。",
            "expected_context_topics": ["格力空调 1.5匹", "CAT-HOME-003", "新一级能效", "整机 6 年保修", "压缩机 10 年包修"],
            "difficulty": "medium",
        },
        {
            "id": "home_robot_001",
            "category": "家用电器",
            "question": "小米扫地机器人 X20 Pro 的价格和功能？",
            "expected_keywords": ["2499", "自动集尘", "热水洗拖布", "5000Pa", "LDS 激光导航"],
            "forbidden_keywords": ["1499", "随机碰撞"],
            "expected_answer": "小米扫地机器人 X20 Pro（商品编号 CAT-HOME-008）售价为 ¥2499 元，配备 LDS 激光导航，精准建图，5000Pa 强劲吸力，自动集尘基站（30 天免倒垃圾），热水洗拖布基站（55℃ 热水清洁，自动烘干拖布）。支持小爱同学语音控制，米家 APP 远程控制，支持多楼层地图保存，自动回充、断点续扫。",
            "expected_context_topics": ["小米扫地机器人 X20 Pro", "CAT-HOME-008", "自动集尘", "热水洗拖布"],
            "difficulty": "medium",
        },
        {
            "id": "home_dyson_hd15_001",
            "category": "家用电器",
            "question": "戴森吹风机 HD15 的价格和特点？",
            "expected_keywords": ["2999", "高速", "防飞翘", "5 档风温", "负离子", "戴森 HD15"],
            "forbidden_keywords": ["1999", "普通吹风机"],
            "expected_answer": "戴森吹风机 HD15（商品编号 CAT-HOME-014）售价为 ¥2999 元，搭载戴森第九代数码马达，转速高达 11 万转/分钟，Air Multiplier 气流倍增技术，产生高压高速气流，智能温控技术防止过热损伤发质，防飞翘风嘴设计。提供 5 档风温（100℃/80℃/60℃/28℃/冷风）、3 档风速，负离子技术减少静电毛躁。",
            "expected_context_topics": ["戴森吹风机 HD15", "CAT-HOME-014", "第九代数码马达", "防飞翘"],
            "difficulty": "hard",
        },
        {
            "id": "home_supor_001",
            "category": "家用电器",
            "question": "苏泊尔 4L 电饭煲的价格和特点是什么？",
            "expected_keywords": ["499", "IH 电磁加热", "球釜内胆", "1200W", "苏泊尔"],
            "forbidden_keywords": ["299", "普通加热盘"],
            "expected_answer": "苏泊尔电饭煲 4L（商品编号 CAT-HOME-015）售价为 ¥499 元，配备 IH 电磁环绕加热，米饭受热均匀，球釜 2.0 内胆聚能锁温，1200W 大火力精控火候。支持多种煮饭模式（柴火饭、快煮、稀饭、锅巴饭等），24 小时预约定时，保温功能最长 12 小时，可拆卸上盖清洗方便。",
            "expected_context_topics": ["苏泊尔电饭煲 4L", "CAT-HOME-015", "IH 电磁加热", "球釜内胆"],
            "difficulty": "medium",
        },

        # ===== 服饰鞋包产品（easy）=====
        {
            "id": "cloth_nike_001",
            "category": "服饰鞋包",
            "question": "Nike Air Force 1 '07 经典款的价格是多少？",
            "expected_keywords": ["899", "Air Force 1", "全粒面皮革", "橡胶外底", "AF1"],
            "forbidden_keywords": ["1299", "网面"],
            "expected_answer": "Nike Air Force 1 '07（商品编号 CAT-CLOTH-001）售价为 ¥899 元，为经典百搭板鞋。采用全粒面皮革鞋面，耐用易清洁，配有 Air-Sole 缓震气垫，轻量舒适，橡胶外底防滑耐磨，低帮设计百搭易穿。耐克官方正品，尺码齐全，支持七天无理由退换。",
            "expected_context_topics": ["Nike Air Force 1", "CAT-CLOTH-001", "经典白色百搭板鞋"],
            "difficulty": "medium",
        },
        {
            "id": "cloth_li_ning_001",
            "category": "服饰鞋包",
            "question": "李宁超轻 21 跑鞋的特点是什么？",
            "expected_keywords": ["799", "䨻中底", "透气", "180g", "超轻 21"],
            "forbidden_keywords": ["专业篮球鞋", "499"],
            "expected_answer": "李宁超轻 21 跑鞋（商品编号 CAT-CLOTH-004）售价为 ¥799 元，采用李宁䨻中底科技，轻弹缓震，䨻丝鞋面技术轻量化透气，CPU 耐磨止滑大底，鞋楦专为亚洲人脚型设计。重量极轻，单只仅约 180g，适合马拉松、日常训练。李宁官方正品。",
            "expected_context_topics": ["李宁超轻21跑鞋", "CAT-CLOTH-004", "轻弹科技", "透气网面"],
            "difficulty": "medium",
        },

        # ===== 食品生鲜产品（easy/medium）=====
        {
            "id": "food_maotai_001",
            "category": "食品生鲜",
            "question": "茅台飞天 53 度的价格是多少？",
            "expected_keywords": ["1499", "500ml", "酱香型", "飞天 53 度", "防伪溯源"],
            "forbidden_keywords": ["999", "清香型"],
            "expected_answer": "茅台飞天 53 度（商品编号 CAT-FOOD-007）售价为 ¥1499 元，为 500ml 单瓶装，酱香型白酒，纯粮固态发酵，四年以上贮存。原厂包装，附防伪溯源码，需实名认证购买，每人每期限量。高端白酒，限量供应，价格以实际下单时为准，提供正规发票。",
            "expected_context_topics": ["茅台飞天53度", "CAT-FOOD-007", "酱香型", "1499 元"],
            "difficulty": "medium",
        },
        {
            "id": "food_crayfish_001",
            "category": "食品生鲜",
            "question": "阳澄湖大闸蟹礼盒的规格和价格？",
            "expected_keywords": ["388", "公蟹 4 两", "母蟹 3 两", "顺丰冷链", "阳澄湖"],
            "forbidden_keywords": ["公蟹 6 两", "免费配送"],
            "expected_answer": "阳澄湖大闸蟹礼盒（商品编号 CAT-FOOD-001）售价为 ¥388 元，为公蟹 4 两/只，母蟹 3 两/只，共 8 只装。阳澄湖原产地直供，顺丰冷链速运，全程 0-8℃ 保鲜，附原产地防伪标签。最佳食用季节为 9-11 月。",
            "expected_context_topics": ["阳澄湖大闸蟹礼盒", "CAT-FOOD-001", "公4两母3两", "冷链运输"],
            "difficulty": "easy",
        },

        # ===== 美妆护肤产品（medium/hard）=====
        {
            "id": "beauty_sk2_001",
            "category": "美妆护肤",
            "question": "SK-II 神仙水 230ml 的价格和功效？",
            "expected_keywords": ["1370", "PITERA", "230ml", "改善肤质", "提亮肤色", "收缩毛孔"],
            "forbidden_keywords": ["680", "去皱抗衰"],
            "expected_answer": "SK-II 护肤精华露（神仙水）230ml（商品编号 CAT-BEAUTY-001）售价为 ¥1370 元，富含超过 90% 的 PITERA 精华，源自天然酵母发酵。功效包括改善肤质、提亮肤色、收缩毛孔。日本原装进口，适合各种肤质，每日早晚洁面后使用。开封后建议 6 个月内用完。正品保障，支持专柜验货。",
            "expected_context_topics": ["SK-II 神仙水", "CAT-BEAUTY-001", "PITERA 酵母精华", "230ml"],
            "difficulty": "medium",
        },
        {
            "id": "beauty_lauder_001",
            "category": "美妆护肤",
            "question": "雅诗兰黛小棕瓶 50ml 的价格和功效？",
            "expected_keywords": ["680", "小棕瓶", "第七代", "Chronolux CB", "夜间修护", "二裂酵母"],
            "forbidden_keywords": ["299", "美白精华"],
            "expected_answer": "雅诗兰黛特润修护肌透精华露（小棕瓶）第七代 50ml（商品编号 CAT-BEAUTY-002）售价为 ¥680 元，独家 Chronolux CB 技术夜间修护肌肤，二裂酵母发酵产物抗氧化抗老。美国品牌原装进口，适合各种肤质，尤其适合有初老、细纹、暗沉问题的肌肤。每日早晚使用于爽肤水之后，开封后 12 个月内用完。",
            "expected_context_topics": ["雅诗兰黛小棕瓶", "CAT-BEAUTY-002", "第七代修护精华", "抗老淡纹"],
            "difficulty": "medium",
        },
        {
            "id": "beauty_lamer_001",
            "category": "美妆护肤",
            "question": "海蓝之谜面霜 60ml 的价格和核心成分是什么？",
            "expected_keywords": ["1999", "Miracle Broth", "神奇活性精粹", "深海巨藻", "修护"],
            "forbidden_keywords": ["999", "玻尿酸"],
            "expected_answer": "La Mer 精华面霜 60ml（商品编号 CAT-BEAUTY-004）售价为 ¥1999 元，核心成分为神奇活性精粹 Miracle Broth，源自深海巨藻发酵。强效修护，改善干燥、敏感、泛红等肌肤问题。美国高端护肤品牌，需先在指间乳化后按压于面部。正品保障，密封包装，支持验货，开封后 24 个月内用完。",
            "expected_context_topics": ["海蓝之谜面霜", "CAT-BEAUTY-004", "经典修护", "深海巨藻"],
            "difficulty": "easy",
        },
    ]


def get_test_cases_by_category(category: Optional[str] = None) -> List[Dict]:
    """按类别筛选测试用例"""
    cases = get_default_test_cases()
    if category is None:
        return cases
    return [c for c in cases if c.get("category") == category]


def list_categories() -> List[str]:
    """返回所有类别（按首次出现顺序）"""
    seen = set()
    result = []
    for c in get_default_test_cases():
        cat = c.get("category", "未分类")
        if cat not in seen:
            seen.add(cat)
            result.append(cat)
    return result
