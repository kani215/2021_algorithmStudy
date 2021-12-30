def check_input(x_data, y_data):
    """check user's input data

    Args:
        x_data ([list]): data list
        y_data ([list]): data list

    Returns:
        [boolean]: if num of each data is same, keep going 
    """
    if len(x_data) != len(y_data):
        return False
    else:
        return True


def costfunction(w, x, y):
    """cost function

    Args:
        w ([double]): this turn 's gradient
        x ([double]): one of x data 
        y ([double]): one of y data 

    Returns:
        ([double]): cost function value
    """
    y_pred = w * x
    return (y_pred - y) ** 2
def gradient(w, x, y):
    """ dericative of cost function about w

    Args:
        w ([double]): this turn 's gradient
        x ([double]): one of x data 
        y ([double]): one of y data 

    Returns:
        [double]: dericative of cost function about w
    """
    return 2 * x * (x * w - y)


def solver(x_data, y_data, learning_rate):
    """solver to find w

    Args:
        x_list ([list]): base x data group
        y_list ([list]): base x data group

    Returns:
        w [double]: optimized w value
    """
    w = 0
    # x_data = [1.0, 2.0, 3.0]
    # y_data = [2.0, 4.0, 6.0]
    if check_input(x_data, y_data) == False:
        return "error"

    for epoch in range(10000):
        grad = 0
        error = 0
        for x_val, y_val in zip(x_data, y_data):            
            temp_grad = gradient(w, x_val, y_val)
            w = w - learning_rate * temp_grad
            print('\tgrad : ', round(temp_grad, 2))
            temp_error = costfunction(w, x_val, y_val)           
            error += temp_error
            grad += temp_grad

        print('progress : ', epoch, 'w = ', round(w, 4), 'error = ', round(error, 4))
        if error < 0.0001:
            print("done")
            return w
