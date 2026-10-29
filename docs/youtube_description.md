# YouTube title and description (draft, 2026-10-29)

Title: PAI最終課題_142340

Description:

Physical AI 応用編1(松尾・岩澤研究室)の最終課題の動画です。
小型二足歩行ロボット Open Duck Mini v2 に、ダンスなどの動きをさせました。すべてシミュレーター(MuJoCo)上で、実機は使っていません。

■ 内容
※ 一部の場面は 2〜4 倍速です(動画内にも表示)。
強化学習(PPO)で歩行を学習し、学習済みの歩きは変えずに、上に「速度指令を毎回計算する制御」を重ねて新しい動きをさせました。
0:00 タイトル
0:06 手本をまねて歩く
0:14 前進の報酬を 2 倍にしたら、歩き始めが早くなった
0:26 報酬 2 倍で長く学習すると、後退・旋回・横歩きもできた
0:42 道案内役: 学習済みの歩きに「行き先の指令」を出す
0:52 サッカー: 道案内役を発展させて、ボールをゴールへ押す
1:07 ダンス(1 曲目)
1:28 ダンス(2 曲目)
2:04 できなかったこと(スキップ風の歩き、サッカーでの転倒、ジャンプ)
2:30 使ったもの・音楽

■ コード・実験の記録
GitHub: https://github.com/wideflat/matsuolab-pai-duck

■ 使ったもの
・Open Duck Mini v2 / Open Duck Project
  https://github.com/apirrone/Open_Duck_Playground
  https://github.com/apirrone/Open_Duck_reference_motion_generator
・MuJoCo / MuJoCo Playground(Google DeepMind)
・Brax(PPO)

■ 音楽
"Fluffing a Duck" Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0 License
http://creativecommons.org/licenses/by/4.0/

"Monkeys Spinning Monkeys" Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0 License
http://creativecommons.org/licenses/by/4.0/

Notes (course rules, 最終課題 PDF): visibility must be at least 限定公開 (unlisted); 公開 is recommended and earns extra credit for 優秀生 selection.
Set 子ども向け = いいえ, 年齢制限 = いいえ; do not tick instant premiere; do not delete or make private until grading is done.
Speed-ups (x4, x2) are stated inside the video captions; add "一部 2〜4 倍速" to the description if wanted.
