import psutil

# 시스템의 물리적 메모리 양 가져오기
total_memory = psutil.virtual_memory().total

# 사용 가능한 물리적 메모리 양 가져오기
available_memory = psutil.virtual_memory().available

# 사용 중인 메모리 양 가져오기
used_memory = psutil.virtual_memory().used

print(f"Total Memory: {total_memory / (1024 * 1024)} MiB")
print(f"Available Memory: {available_memory / (1024 * 1024)} MiB")
print(f"Used Memory: {used_memory / (1024 * 1024)} MiB")
