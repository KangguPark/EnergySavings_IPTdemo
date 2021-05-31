PALETTE = ['#F4AB33', '#EF157B', '#EE1C25', '#323494', '#00AEDE', '#814f37', '#444444']
SOLARIZED = ['#B58900', '#CB4B16', '#DC322F', '#D33682', '#6C71C4', '#268BD2', '#2AA198', '#859900']

def lim_margin(l, u, margin=0.05):
    b = u - l
    return l - margin*b, u + margin*b