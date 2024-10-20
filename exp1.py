import statistics

# 1. Take input from user (list elements)
numbers = list(map(int, input("Enter list elements separated by spaces: ").split()))

# 2. Find the size of the list
print("Size of the list:", len(numbers))

# 3. Sort the list
sorted_list = sorted(numbers)
print("Sorted list (ascending):", sorted_list)

# 4. Calculate mean, median, and mode
print("Mean:", statistics.mean(numbers))
print("Median:", statistics.median(numbers))
print("Mode:", statistics.mode(numbers) if len(set(numbers)) != len(numbers) else "No unique mode")

# 5. Calculate variance and standard deviation
print("Variance:", statistics.variance(numbers))
print("Standard Deviation:", statistics.stdev(numbers))
