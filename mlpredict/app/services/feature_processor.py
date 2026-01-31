class FeatureProcessor:
    def __init__(self):
        # 特征映射字典
        self.toilet_map = {
            '否': 0,
            '是': 1
        }
        
        self.toilet_score = {
            '传统旱厕': 0,
            '冲洗坑厕': 1,

            '抽水马桶': 2,
        }
        
        self.house_score = {
            '自己购买新房': 4,
            '自己购买二手房': 3,
            '自建房': 2,
            '租房': 1,
            '否': 0
        }
        
        self.freq_map = {
            '否': 0,
            '1-2次/年': 1,
            '1-2次/月': 2,
            '1-2次/周': 3,
            '3-5次/周': 4,
            '＞5次/周': 5,
            '未填': None
        }
        
        self.veg_score = {
            '自家种植': 0,
            '超市': 1,
            '菜市场': 2,
            '街头小贩': 3,
            '都有': 3
        }
    
    def process_toilet_lid(self, value):
        """处理马桶盖使用习惯"""
        if value in ['抽水马桶', '未填']:
            return None
        return self.toilet_map.get(value, None)
    
    def process_toilet_type(self, value):
        """处理家庭厕所类型"""
        parts = value.replace(' ', '').split('+')
        return max(self.toilet_score.get(p, 0) for p in parts)
    
    def process_house_ownership(self, value):
        """处理居住房屋所有权"""
        return self.house_score.get(value, 0)
    
    def process_snack_frequency(self, value):
        """处理零食的食用频率"""
        return self.freq_map.get(value, None)
    
    def process_vegetable_purchase(self, value):
        """处理家中蔬菜的购买方式"""
        parts = value.replace(' ', '').split('+')
        return max(self.veg_score.get(p, 0) for p in parts)
    
    def process_all_features(self, features):
        """处理所有特征
        
        Args:
            features: 特征字典，包含所有需要处理的特征
            
        Returns:
            list: 处理后的特征值列表，按照指定顺序
        """
        processed_features = []
        
        # 按照指定顺序处理特征
        # 1. 如果使用马桶，是否习惯盖马桶盖
        processed_features.append(self.process_toilet_lid(features.get('toilet_lid', '')))
        
        # 2. 家庭厕所类型
        processed_features.append(self.process_toilet_type(features.get('toilet_type', '')))
        
        # 3. 居住房屋所有权
        processed_features.append(self.process_house_ownership(features.get('house_ownership', '')))
        
        # 4. 零食的食用频率
        processed_features.append(self.process_snack_frequency(features.get('snack_frequency', '')))
        
        # 5. 家中蔬菜的购买方式
        processed_features.append(self.process_vegetable_purchase(features.get('vegetable_purchase', '')))
        
        return processed_features