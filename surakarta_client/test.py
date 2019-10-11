from excavator import energy


if __name__ == '__main__':
    e = energy.Energy()
    print(e.get_max_rate_step(1, 24))