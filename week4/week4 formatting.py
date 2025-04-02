# 기본 사용법
name = "Alice"
age = 30

print(f'제 이름은 {name}이고, 나이는{age}세입니다.')

# 표현식 포함
x = 10
y = 20
print(f"x + y의 합은 {x+y}입니다.")

# 형식 지정
price = 1000
print(f"가격은 {price:,}원 입니다.") #천 단위 구분 쉼표

# 소수점
pi = 3.14159
print(f"원주율은 {pi:.2f}입니다.") #소수점 2자리까지

# 메서드 호출
text = "Hello World"
print(f"대문자: {text.upper()}")
print(f"소문자: {text.lower()}")

# 정렬과 폭 지정
name = "Bob"
print(f"{'<'}{name:^15}{'>'}") #가운데 정렬, 총 10자리