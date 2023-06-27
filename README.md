# symbolic-tod
Public repo for symbolic dialogue state tracking and potentially next action prediction

format:
[tasks]
[tags]
[params] p0=[slot0 desc] p1=[slot1 desc]...
[useracts] u0=[useraction0] u1=[useraction1]... u4=out of doamin
[sysacts] s0=[systemaction0] s1=[systemaction1]... s5=query data base s6=out of domain
[dependencies] u2,u3->s5; u2,s3->s6
[target actions] s5
[constraints] user request pi -> system inform pi; target action depend on pi -> system request pi
[conversation] [user] utterance [system] utterance... \n

[states] p_i=[value_i] p_j=[value_j]...
[history] u_i u_k; s_j;...
[next action] s3 s4 s1