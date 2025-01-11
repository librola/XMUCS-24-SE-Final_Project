def tests_filter(path: str):
    # 应该满足 python 文件所在的目录名为 tests
    return "tests" in path.split("/")
    

def tests_files_and_classes_and_functions_in_(files, classes, functions):