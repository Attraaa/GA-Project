import numpy as np
import json
import os

# 개쩌는 원리 정리
# 컨테이너의 높이는 생각하지 않고 일단 width를 정의함 ex) 10
# 0으로된 배열을 만듦 np.zeros(container_width)
# 6x4의 박스를 0번째 인덱스에 넣는다면?
# [4, 4, 4, 4, 4, 4, 0, 0, 0, 0] 이렇게 높이를 저장함
# 만약 3x3을 또 넣는다면? 가장 낮은곳을 찾아서 들어갈 수 있는 공간이면 쌓아버림
# [4, 4, 4, 4, 4, 4, 3, 3, 3, 0] 이렇게 되어버림
# 이제 가장 높은 곳에서 바닥까지의 넓이에서 모든 박스의 넓이를 빼주면 빈 공간이 나옴
# 만약 컨테이너의 높이를 정한다면 다 쌓았을때 컨테이너의 높이를 넘어가는 유전자에 대해 큰 페널티를 주면 됨


# 만약 박스를 다 쌓았는데 [4, 4, 4, 4, 4, 4, 3, 3, 3, 0] 이랑 [4, 3, 3, 3, 3, 3, 3, 3, 3, 0]
# 뭐 이런식으로 쌓였다? 그러면 전자보다 후자의 경우가 더 잘 쌓았다고 볼 수 있을거임 근데 앞서나온 방법으로는 판단할 수 없음 (최대 높이만 고려하기 때문에 둘다 4로 동일함)
# 그래서 각 인덱스마다 최대 높이를 뺀 값을 모두 더해주면
# [4-4 4-4, 4-4, 4-4, 4-4, 4-4, 4-3, 4-3, 4-3, 4-0] vs [4-4, 4-3, 4-3, 4-3, 4-3, 4-3, 4-3, 4-3, 4-3, 4-0]
# [0, 0, 0, 0, 0, 0, 1, 1, 1, 4] = 7  vs [0, 1, 1, 1, 1, 1, 1, 1, 1, 4] = 12
# 이런식으로 후자를 점수를 더 높게 주면 될 것 같음

# 로직 고민하다가 이게 맞나 싶어서 일단 정리해봤는데 틀린점 있으면 말좀 (톡에 보내준 이미지 참고)

# 참고사항
# 유전자는 박스 쌓는 순서와 방향만 갖고 있음
# 쌓는것은 "가장 낮은곳을 찾아서 들어갈 수 있는 공간이면 쌓아버림"으로 고정되어 있기 때문에 배치하는 쌓는 위치를 기억할 필요가 없음
# 근데 더 중요한것은 이 알고리즘이 틀렸을 수도 있다는 거임 그럼 줴줴이야~
# https://tibyte.kr/239/ 이 글이 도움이 될 듯




#####################################config#####################################
result_folder = 'results'
os.makedirs(result_folder, exist_ok=True)

container_width = 25  # 컨테이너 너비

# 박스 가로 세로 (연구소아님) 개수 딕셔너리로 만들기
boxes = [
    {"w": 2, "h": 3, "count": 7},
    {"w": 2, "h": 4, "count": 9},
    {"w": 3, "h": 3, "count": 8},
    {"w": 4, "h": 4, "count": 7},
    {"w": 5, "h": 5, "count": 9},
    {"w": 6, "h": 6, "count": 6},
]

# 모집단 크기
population_size = 1000

#부모 후보 수
# parents = 50

# 세대 수
generation_count = 50

# 돌연변이 확률
mutation_rate = 0.05

#####################################config#####################################





# 박스 클래스
class Box:
    def __init__(self, id, width, height):
        self.id = id
        self.width = width
        self.height = height

    def rotated(self):
        return Box(self.id, self.height, self.width)


# 한바꾸 돌면 여기에 저장됨
# gene 예시
# gene = [
# (Box2(2x6), True),     #(Box 객체, 회전 여부) 배치된 순서대로 저장되어 있음 아마도..
# (Box0(3x3), False),
# (Box3(3x9), True),
# (Box1(3x3), False)
# ]
class Individual:
    def __init__(self, gene):
        self.gene = gene
        self.fitness = 0.0  # 점수 (높을 수록 좋음)
        self.area_sum=0
        self.max_height = 0
        self.wasted_space = 0


    def calculate_fitness(self):
        container = np.zeros(container_width)
        total_box_area = 0

        for (original_box, is_rotated) in self.gene:
            box = None
            if is_rotated:
                box = original_box.rotated()
            else:
                box = original_box
            box_h = box.height
            box_w = box.width

            total_box_area += box_h * box_w

            best_x=-1
            min_top_y = float('inf')


            #자 이거 설명해줌
            #현재 순서에서 사용될 박스를들고 가장 왼쪽부터 순회돌기
            #가장 낮은 위치가 되는 x좌표를 찾기
            #그리고 그 [x좌표:+x좌표+박스너비] 구간에 현재 박스의 높이를 더한 값을 저장

            for x in range (container_width - box_w + 1):
                base_y = np.max(container[x: x + box_w])
                current_top_y = base_y + box_h

                if current_top_y < min_top_y:
                    min_top_y = current_top_y
                    best_x = x

            if best_x!=-1: container[best_x: best_x + box_w] = min_top_y

        self.max_height = np.max(container)
        self.area_sum = np.sum(container)

        penalty = (self.max_height * 100) + self.area_sum
        self.fitness = (1.0 / penalty)*10000

        self.wasted_space =  self.area_sum - total_box_area

        # Individual 클래스 내부 메서드로 추가

    def copy(self):
        # 유전자 리스트를 슬라이싱[:]해서 새로운 리스트 객체로 만듦 (깊은 복사 효과)
        new_gene = self.gene[:]
        new_ind = Individual(new_gene)

        # 계산된 값들도 그대로 복사
        new_ind.fitness = self.fitness
        new_ind.area_sum = self.area_sum
        new_ind.max_height = self.max_height
        new_ind.wasted_space = self.wasted_space
        return new_ind




class GeneticAlgorithm:
    def __init__(self, box_data_list, pop_size=population_size, _mutation_rate = mutation_rate):
        self.pop_size = pop_size
        self.mutation_rate = _mutation_rate
        self.base_boxes = []
        self.population = []

        current_id = 0
        for data in box_data_list:
            for _ in range(data["count"]):
                new_box = Box(current_id, data["w"], data["h"])
                self.base_boxes.append(new_box)
                current_id += 1


    def init_population(self):
        for i in range(population_size):
            gene = [] #박스 객체랑 순서 저장
            boxes_copy = self.base_boxes.copy()
            np.random.shuffle(boxes_copy)

            for box in boxes_copy:
                is_rotated = np.random.choice([True, False])
                gene.append( (box, is_rotated) )

            individual = Individual(gene)
            individual.calculate_fitness()
            self.population.append(individual)#모집단에 추가

    # def select_parent(self,parents):
    #     candidates = np.random.choice(self.population, size=parents, replace=False)
    #     return max(candidates, key=lambda ind: ind.fitness) #후보 중 가장 우수한 개체 하나만

    def swap_mutation(self, individual):

        if np.random.random() > self.mutation_rate:
            return

        #순서 바꾸기
        idx1, idx2 = np.random.choice(range(len(individual.gene)), 2)
        individual.gene[idx1], individual.gene[idx2] = individual.gene[idx2], individual.gene[idx1]

        #rotate 바꾸기
        if np.random.random() < 0.5:
            box_obj, is_rotated = individual.gene[idx1]
            individual.gene[idx1] = (box_obj, not is_rotated)

    def order_crossover(self, parent1, parent2):
        # 유전 알고리즘 교차 정리 참고 http://www.aistudy.com/biology/genetic/operator_moon.htm
        # 우리는 염색체가 순열로 표시되기 때문에 Order Crossover, PMX중 사용해야험
        # =========원리정리=========
        # 두 개의 자름선을 임의로 뽑아 부모1의 유전자를 자르고 그걸 자식한테 그대로 줌.
        # 그리고 부모1한테 물려받은 부분을 제외한 나머지 부분(중복되면 안되니까)을 부모2를 순회하면서 순서대로 넣음됨
        # 근데 이게 최선인진 모르겠음.. 바닥부터 쌓으니까 높은 점수의 유전자는 배열의 앞부분이 더 중요할거같은데(추측) 이건 자름선이 완전 랜덤이라 과연 의도한 대로 좋은 자식 유전자를 만들 수 있을까?? 그래서 엘리티즘이 있는건가? ㅁㄹ
        size = len(parent1.gene)
        child_gene = [None] * size

        # 부모1길이 만큼의 인덱스에서 랜덤한 두 수를 뽑아 정렬 후 부모1 슬라이싱해서 자식한테 그 인덱스 그 자리 그대로 물려줌
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child_gene[start : end + 1] = parent1.gene[start : end + 1]

        # 자식한테 이미 들어간 박스들의 ID를 기록해서 중복 방지
        copied_ids = {item[0].id for item in child_gene[start : end + 1]}

        # 부모2 유전자를 처음부터 순회하면서 자식에게 없는 박스만 빈칸에 순서대로 채워 넣음
        current_idx = 0
        for gene_item in parent2.gene:
            box_id = gene_item[0].id

            if box_id in copied_ids:
                continue

            while child_gene[current_idx] is not None:
                current_idx += 1

            child_gene[current_idx] = gene_item

        return Individual(child_gene)

    #상위 num_individuals개 만큼 뽑기
    def elitism_selection(self, num_individuals):
        individuals = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        return individuals[:num_individuals]


    def tournament_selection(self, num_individuals, tournament_size=5):
        selected = []

        for _ in range(num_individuals):
            candidates = np.random.choice(self.population, size=min(tournament_size, self.pop_size), replace=False)
            best = max(candidates, key=lambda ind: ind.fitness)
            selected.append(best)

        return selected

    def evolve(self, elite_count=5, tournament_size=5):

        next_generation = []

        elites = self.elitism_selection(elite_count)


        for elite in elites:
            next_generation.append(elite.copy())


        while len(next_generation) < self.pop_size:
            # (1) 부모 선택 (Selection): 토너먼트 방식으로 아빠, 엄마 선정
            parent1 = self.tournament_selection(1, tournament_size)[0]
            parent2 = self.tournament_selection(1, tournament_size)[0]

            child = self.order_crossover(parent1, parent2)
            self.swap_mutation(child)
            child.calculate_fitness()

            next_generation.append(child)


        self.population = next_generation


#나중에 한번 봐야할듯 시각화
def get_placement_info(individual):
    container = np.zeros(container_width)
    placements = []  # {id, x, y, w, h, rotated}

    for (original_box, is_rotated) in individual.gene:
        box = original_box.rotated() if is_rotated else original_box
        box_w, box_h = box.width, box.height

        best_x = -1
        min_top_y = float('inf')
        base_y_at_best_x = 0

        for x in range(container_width - box_w + 1):
            base_y = np.max(container[x: x + box_w])
            current_top_y = base_y + box_h

            if current_top_y < min_top_y:
                min_top_y = current_top_y
                best_x = x
                base_y_at_best_x = base_y 

        if best_x != -1:
            container[best_x: best_x + box_w] = min_top_y

            placements.append({
                "id": original_box.id,
                "x": int(best_x),
                "y": int(base_y_at_best_x),  # 바닥 높이
                "w": int(box_w),
                "h": int(box_h),
                "is_rotated": bool(is_rotated)
            })

    return placements

if __name__ == "__main__":
    print(f"프로젝트 시작: 총 {len(boxes)} 종류의 박스 적재 최적화")
    print(f"설정: 세대 수={generation_count}, 인구 수={population_size}, 변이율={mutation_rate}")

    ga = GeneticAlgorithm(boxes)
    ga.init_population()

    initial_best = max(ga.population, key=lambda ind: ind.fitness)
    print(f"\n[초기 상태] 최고 높이: {initial_best.max_height}, 적합도: {initial_best.fitness:.4f}")

    for i in range(generation_count):

        ga.evolve(elite_count=5, tournament_size=5)

        best_individual = max(ga.population, key=lambda ind: ind.fitness)


        #JSON 저장~
        placements = get_placement_info(best_individual)

        json_data = {
            "generation": i + 1,
            "fitness": float(best_individual.fitness),
            "max_height": float(best_individual.max_height),
            "wasted_space": float(best_individual.wasted_space),
            "container_width": container_width,
            "placements": placements
        }


        file_path = os.path.join(result_folder, f'gen_{i + 1}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)



        print(f"\n========== {i + 1} 세대 ==========")
        print("최고 유전자")
        print(f"적합도 점수 : {best_individual.fitness:.8f}")  # 소수점 8자리까지 보기
        print(f"높이: {best_individual.max_height}")
        print(f"쌓은 면적 합(빈공간 포함, 천장빈공간 미포함): {best_individual.area_sum}")
        print(f"낭비된 공간 : {best_individual.wasted_space}")

        
        if best_individual.wasted_space == 0:
            print("\n최적해 발견.")
            break
    print("\n--- 결과 ---")

    final_best = max(ga.population, key=lambda ind: ind.fitness)
    print(f"최종 최적 높이: {final_best.max_height}")
    print(f"낭비된 공간: {final_best.wasted_space}")
    
# TODO: