from hmm import HMM

hmm = HMM()

hmm.fit(
    observations=[
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100)
    ]
)

print(hmm.predict_rate_limit(
    observations=[
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100),
        (False, 100),
        (True, 100)
    ]
))