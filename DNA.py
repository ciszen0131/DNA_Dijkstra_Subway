import heapq

YIDONG = 2
HWANSEUNG = 5

line_1 = [
    "청량리역", "제기동역", "신설동역", "동묘앞역", "동대문역", "종로5가역", "종로3가역", "종각역",
    "시청역", "서울역", "남영역", "용산역", "노량진역", "대방역", "신길역", "영등포역",
    "신도림역", "구로역", "구일역", "개봉역", "오류동역", "온수역", "역곡역"
]

line_2 = [
    "동대문역사문화공원역", "을지로4가역", "을지로3가역", "을지로입구역", "시청2역", "충정로역",
    "아현역", "이대역", "신촌역", "홍대입구역", "합정역", "당산역", "영등포구청역",
    "문래역", "신도림2역", "대림역", "구로디지털단지역", "신대방역", "신림역"
]

graph = {}

def link(line, line_name):
    for i in range(len(line)):
        if line[i] not in graph:
            graph[line[i]] = {}
        if i > 0:
            graph[line[i]][line[i - 1]] = (YIDONG, line_name)
        if i < len(line) - 1:
            graph[line[i]][line[i + 1]] = (YIDONG, line_name)

link(line_1, "1호선")
link(line_2, "2호선")

graph["시청역"]["시청2역"] = (HWANSEUNG, "환승")
graph["시청2역"]["시청역"] = (HWANSEUNG, "환승")
graph["신도림역"]["신도림2역"] = (HWANSEUNG, "환승")
graph["신도림2역"]["신도림역"] = (HWANSEUNG, "환승")

def dijkstra(start, end):
    distances = {}
    for node in graph:
        distances[node] = float('inf')

    previous = {}
    for node in graph:
        previous[node] = None

    line_info = {}
    for node in graph:
        line_info[node] = None

    distances[start] = 0
    queue = [(0, start)]
    
    while queue:
        dist, now = heapq.heappop(queue)
        
        if dist > distances[now]:
            continue
        
        for neighbor, (weight, line) in graph[now].items():
            distance = dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = now
                line_info[neighbor] = line
                heapq.heappush(queue, (distance, neighbor))
    
    path = []
    lines = []
    node = end
    while node:
        path.append(node)
        lines.append(line_info[node])
        node = previous[node]
    
    path.reverse()
    lines.reverse()
    return distances[end], path, lines

def rename(name):
    return name.replace("2역", "역") if "2역" in name else name

start = input("출발역을 입력하세요: ")
end = input("도착역을 입력하세요: ")

if start not in graph or end not in graph:
    print("입력하신 역 이름이 올바르지 않습니다.")
else:
    time, path, lines = dijkstra(start, end)

    print(f"{rename(path[0])}에서 탑승해서 ", end="")

    for i in range(len(path) - 1):
        # 두 역이 다른 노선에 속하는 경우에만 환승 시간 추가
        if lines[i] != lines[i + 1] and lines[i + 1] != "환승":
            print(f"{rename(path[i])}에서 {lines[i + 1]}으로 환승하신 다음에 ", end="")

    print(f"{rename(path[-1])}에서 내리시면 됩니다.")
    if ((start == "신도림역") or (start == "시청역")):
        if (end in line_2):
            time -= HWANSEUNG
        

    if ((end == "신도림역") or (end == "시청역")):
        if (start in line_2):
            time -= HWANSEUNG
        
    
    print(f"최단 시간: {time}분")
