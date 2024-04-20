def colorArrayResize(arr, target_len):
    def avg(arr):
        return [sum(x) / len(x) for x in zip(*arr)]

    if len(arr) == target_len:
        # No need to do anything
        return arr
    elif len(arr) > target_len:
        # Shrink the array, averaging the colors of adjacent pixels, until the target length is reached
        while len(arr) > target_len:
            tmp_arr = []
            for i in range(0, len(arr) - 1, 2):
                tmp_arr.append(avg([arr[i], arr[i + 1]]))
            arr = tmp_arr

        return arr
    else:
        # Expand the array, creating new pixels by averaging the colors of adjacent pixels, until the target length is reached
        while len(arr) < target_len:
            tmp_arr = []
            for i in range(len(arr)):
                tmp_arr.append(arr[i])
                if i + 1 < len(arr):
                    tmp_arr.append(avg([arr[i], arr[i + 1]]))
            arr = tmp_arr

        return arr