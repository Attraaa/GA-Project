import numpy as np

# 개쩌는 원리 정리
# 컨테이너의 높이는 생각하지 않고 일단 width를 정의함 ex) 10
# 0으로된 배열을 만듦 np.zeros(container_width)
# 6x4의 박스를 0번째 인덱스에 넣는다면?
# [4, 4, 4, 4, 4, 4, 0, 0, 0, 0] 이렇게 높이를 저장함
# 만약 3x3을 또 넣는다면? 가장 낮은곳을 찾아서 들어갈 수 있는 공간이면 쌓아버림
# [4, 4, 4, 4, 4, 4, 3, 3, 3, 0] 이렇게 되어버림
# 이제 가장 높은 곳에서 바닥까지의 넓이에서 모든 박스의 넓이를 빼주면 빈 공간이 나옴
# 만약 컨테이너의 높이를 정한다면 다 쌓았을때 컨테이너의 높이를 넘어가는 유전자에 대해 ㅈㄴ큰 페널티를 주면 됨


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


# 교수님도 가중치 디렉토리를 여기에 정의하셨음 우리도 나중에 폴더 만들어서 저장하죠
data_folder = 'weights'

container_width = 10  # 컨테이너 너비

# 박스 가로 세로 (연구소아님) 개수 딕셔너리로 만들기
boxes = [
    {"w": 2, "h": 3, "count": 15},
    {"w": 2, "h": 4, "count": 15},
    {"w": 3, "h": 3, "count": 12}
]

#모집단 50000개로 해봤는데 지금 boxes 기준으로 최고 점수: 0.3030273278529023 나옴 세대 돌리면 더 올라갈라나?
population_size = 50000 #모집단 크기

generation_count = 50 #세대 수

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

        penalty = (self.max_height * 1000000) + self.area_sum
        self.fitness = (1.0 / penalty)*10000000

        self.wasted_space =  self.area_sum - total_box_area
#-----------------------------------------------------------------------
    #copy함수 생성. 독립개체  생성
    def copy(self):
        new_gene = [(box, rot) for (box, rot) in self.gene]
        return Individual(new_gene)
#-----------------------------------------------------------------------
    




class GeneticAlgorithm:
    def __init__(self, box_data_list, pop_size=population_size):
        self.pop_size = pop_size
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
#-----------------------------------------------------------------
    def eval_population(self):
        for ind in self.population:
            ind.calculate_fitness()
    def select_parent(self,k=5):
        candidates = np.random.choice(self.population, size=k, replace=False)
        return max(candidates, key=lambda ind: ind.fitness)
    

    def swap_mutation(individual, mutation_rate=0.02):

        if np.random.rand() > mutation_rate:
            return individual

        gene = individual.gene
        length = len(gene)
        if length < 2:
            return individual

        a, b = np.random.choice(length, 2, replace=False)
        gene[a], gene[b] = gene[b], gene[a]

        return individual

    def orderCrossover(parent1:Individual,parent2:Individual):
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        size = len(offspring1.gene)

        child = [None]*size
        start, end = sorted(np.random.choice(size, 2, replace=False))

        child[start:end+1] = offspring1[start:end+1]

        pos = (end + 1) % size
        for item in offspring2:
            if item not in child:
                child[pos] = item
                pos = (pos + 1) % size

        offspring = parent1.copy()
        offspring.gene = child
        return offspring

    #엘리트 뽑을때 copy사용해야함
    def elitism_selection(population, num_individuals):
        individuals = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        return individuals[:num_individuals]

    def tournament_selection(population, num_individuals, tournament_size=5):
        selected = []
        pop_size = len(population)

        for _ in range(num_individuals):
            candidates = np.random.choice(population, size=min(tournament_size, pop_size), replace=False)
            best = max(candidates, key=lambda ind: ind.fitness)
            selected.append(best)

        return selected
    def evolve(self, elite_count=5, tournament_size=5, mutation_rate=0.02):
            new_population = []

            elites = self.elitism_selection(self.population, elite_count)
            new_population.extend([e.copy() for e in elites])

            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(self.population, 1, tournament_size)[0]
                p2 = self.tournament_selection(self.population, 1, tournament_size)[0]

                child = self.orderCrossover(p1, p2)

                if np.random.rand() < mutation_rate:
                    self.swap_mutation(child, mutation_rate)

                child.calculate_fitness()
                new_population.append(child)

            self.population = new_population
#-----------------------------------------------------------------
        

ga = GeneticAlgorithm(boxes, population_size) 
ga.init_population()
print(f"생성된 개체 수: {len(ga.population)}") 
#10개 미리보기
for i in range(0, 100):
    print(f"{i} 번째 유전자 점수: {ga.population[i].fitness}")
    print(f"    최대 높이: {ga.population[i].max_height}, 쌓은 면적 합(빈공간 포함, 천장빈공간 미포함): {ga.population[i].area_sum}, 낭비된 공간: {ga.population[i].wasted_space}")

best_individual = max(ga.population, key=lambda ind: ind.fitness)
print(f"최고 점수: {best_individual.fitness}")


# TODO:
# 1. 세대 반복
# 2. 선택
# 3. 교차
# 4. 돌연변이


# master_box_list = []
# current_id = 0
# for i in boxes:
#     for j in range(i["count"]):
#         new_box = Box(current_id, i["w"], i["h"])
#         master_box_list.append(new_box)
#         current_id += 1
