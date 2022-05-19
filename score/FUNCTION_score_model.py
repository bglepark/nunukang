# 클래스 생성
class indi_score:
    
    # 속성 생성
    def __init__(self, category_id, fish_size):
        self.category_id = category_id
        self.fish_size = fish_size
        
    def get_score(Percentile_Rank):
        if Percentile_Rank >= 0.99: # 상위 0.1% 이내
            score = 100
        elif 0.90 < Percentile_Rank: # 상위 10% 이내
            score = 90
        elif 0.85 < Percentile_Rank: # 상위 15% 이내
            score = 85
        elif 0.80 <  Percentile_Rank: # 상위 20% 이내
            score = 80
        elif 0.75 < Percentile_Rank: # 상위 25% 이내
            score = 75
        elif 0.70 < Percentile_Rank: # 상위 30% 이내
            score = 70
        elif 0.65 < Percentile_Rank: # 상위 35% 이내
            score = 65
        elif 0.60 < Percentile_Rank: # 상위 40% 이내
            score = 60
        elif 0.55 < Percentile_Rank: # 상위 45% 이내
            score = 55
        elif 0.50 < Percentile_Rank: # 상위 50% 이내
            score = 50
        elif 0.45 < Percentile_Rank: # 상위 55% 이내
            score = 45
        elif 0.40 < Percentile_Rank: # 상위 60% 이내
            score = 40
        elif 0.35 < Percentile_Rank: # 상위 65% 이내
            score = 35
        elif 0.30 < Percentile_Rank: # 상위 70% 이내
            score = 30
        elif 0.25 < Percentile_Rank: # 상위 75% 이내
            score = 25
        elif 0.20 < Percentile_Rank: # 상위 80% 이내
            score = 20
        elif 0.15 < Percentile_Rank: # 상위 85% 이내
            score = 15
        elif 0.10 < Percentile_Rank: # 상위 90% 이내
            score = 10
        elif Percentile_Rank <= 0.10: # 그 이하
            score = 5
        return score
    
    
    # 0. 광어 메소드 생성
    def zero(self, fish_size):
        df_total_광어 = df_total[(df_total['category_id']==0)&(df_total['fish_size'] > 35)]
        df_total_광어['Percentile_Rank'] = df_total_광어.fish_size.rank(pct = True, method='min').round(2)
        df_total_광어['fish_size'] = df_total_광어.fish_size.round(2)
        df_total_광어['score'] = df_total_광어['Percentile_Rank'].apply(lambda Percentile_Rank: get_score(Percentile_Rank))
        return df_total_광어.loc[df_total_광어['fish_size']== fish_size].score

    
    
    # 1. 우럭 메소드 생성
    def one(self, fish_size):
        df_total_우럭 = df_total[(df_total['category_id']==1)&(df_total['fish_size'] > 23)]
        df_total_우럭['Percentile_Rank'] = df_total_우럭.fish_size.rank(pct = True, method='min').round(2)
        df_total_우럭['fish_size'] = df_total_우럭.fish_size.round(2)
        df_total_우럭['score'] = df_total_우럭['Percentile_Rank'].apply(lambda Percentile_Rank: get_score(Percentile_Rank))
        return df_total_우럭.loc[df_total_우럭['fish_size']== fish_size].score
        
        
    # 2. 참돔 메소드 생성
    def two(self, fish_size):
        df_total_참돔 = df_total[(df_total['category_id']==2)&(df_total['fish_size'] > 24)]
        df_total_참돔['Percentile_Rank'] = df_total_참돔.fish_size.rank(pct = True, method='min').round(2)
        df_total_참돔['fish_size'] = df_total_참돔.fish_size.round(2)
        df_total_참돔['score'] = df_total_참돔['Percentile_Rank'].apply(lambda Percentile_Rank: get_score(Percentile_Rank))
        return df_total_참돔.loc[df_total_참돔['fish_size']== fish_size].score
    
    # 3. 감성돔 메소드 생성
    def three(self, fish_size):
        df_total_감성돔 = df_total[(df_total['category_id']==3)&(df_total['fish_size'] > 25)]
        df_total_감성돔['Percentile_Rank'] = df_total_감성돔.fish_size.rank(pct = True, method='min').round(2)
        df_total_감성돔['fish_size'] = df_total_감성돔.fish_size.round(2)
        df_total_감성돔['score'] = df_total_감성돔['Percentile_Rank'].apply(lambda Percentile_Rank: get_score(Percentile_Rank))
        return df_total_감성돔.loc[df_total_감성돔['fish_size']== fish_size].score   
    
    # 4. 돌돔 메소드 생성
    def four(self, fish_size):
        df_total_돌돔 = df_total[(df_total['category_id']==4)&(df_total['fish_size'] > 24)]
        df_total_돌돔['Percentile_Rank'] = df_total_돌돔.fish_size.rank(pct = True, method='min').round(2)
        df_total_돌돔['fish_size'] = df_total_돌돔.fish_size.round(2)
        df_total_돌돔['score'] = df_total_돌돔['Percentile_Rank'].apply(lambda Percentile_Rank: get_score(Percentile_Rank))
        return df_total_돌돔.loc[df_total_돌돔['fish_size']== fish_size].score  

# ========================================================================================== #
# 예시

# indi_score의 광어 score 인스턴스 생성
# score = indi_score(0, 56.1710)

# 광어 score 인스턴스의 zero 메소드 호출
# print(score.zero(56.1710))
# ==> 광어 56.1710cm 에 해당하는 score 값 도출됨