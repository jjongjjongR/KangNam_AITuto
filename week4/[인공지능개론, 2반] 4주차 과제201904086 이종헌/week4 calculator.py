def calculator():
    print("간단한 계산기 프로그램입니다.")
    while True: 
        try:
            num1 = float(input("첫 번째 숫자를 입력해주세요: "))
            operator = input("연산자 (+,-,*,/)를 입력해주세요: ")
            num2 = float(input("두 번째 숫자를 입력해주세요: "))
        
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2            
            elif operator == '*':
                result = num1 * num2
            elif operator == '/':
                if num2 == 0:
                    raise ZeroDivisionError("0으로 나눌 수 없습니다.")
                result = num1 / num2
            else:
                print("잘못된 연산자입니다. 다시 시도해주세요.")
                continue
        
            print(f"결과: {num1} {operator} {num2} = {result}")

        except ValueError:
            print("잘못된 입력입니다. 숫자를 입력해주세요.")
        except ZeroDivisionError as e:
            print(e)
    
        again = input("다시 계산하시겠습니까? (y/n): ").strip().lower() #공백 제거, 소문자
        if again != 'y':
            print("계산기를 종료합니다.")
            break
    
if __name__ == "__main__":
    calculator()
#다른 곳에서 import하면 calculator 실행 안됨.