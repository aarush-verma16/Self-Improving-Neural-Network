# Related work

Audit date: 2026-10-10. A citation below is included because a primary page or PDF was opened in that pass. Entries are in `paper/refs.bib`. Sources that were not opened are listed at the end and are not cited as support.

## Self-modifying weights

Schmidhuber, "A 'Self-Referential' Weight Matrix," ICANN 1993, pages 446–450, DOI 10.1007/978-1-4471-2063-6_107. A recurrent net is given addresses for its own connections and input/output units that read and write those weights, including the weights that do the reading and writing. The paper's claim is possibility, not efficiency. The stated per-step cost is O(n_conn log n_conn), and the error surface is described as full of local minima. The author's copy of the argument is at https://people.idsia.ch/~juergen/selfref/.

Schlag, Irie, and Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers," ICML 2021, PMLR 139, pages 9355–9366. Linearized self-attention is the same object as a 1990s fast-weight controller: a slow net writes a fast key-value memory with outer products. They replace the additive write with a delta-rule update. The experiments are retrieval, translation, and language modeling. This is self-modification inside a forward pass, not a logged architectural edit between training steps.

## Nested learning and HOPE

Behrouz, Razaviyayn, Zhong, and Mirrokni, "Nested Learning: The Illusion of Deep Learning Architectures," NeurIPS 2025. The PDF opened from the NeurIPS proceedings spells the third author Peiling Zhong (https://proceedings.neurips.cc/paper_files/paper/2025/file/4309616aaed8e848009bc4a7ef73b493-Paper-Conference.pdf). An arXiv HTML copy is 2512.24695. Google's research blog summarizes the same paper: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/.

Nested Learning treats a model as several optimization problems, each with its own context flow and update rate. HOPE is their proof-of-concept: a self-modifying Titans-style recurrent model plus a continuum memory, aimed at language modeling, continual learning, and long context. That is a different substrate and a different compute regime from this project. We do not reimplement HOPE, and we do not treat their language-model numbers as a baseline we must match.

## Function-preserving growth

Chen, Goodfellow, and Shlens, "Net2Net: Accelerating Learning via Knowledge Transfer," ICLR 2016, arXiv:1511.05641. Function-preserving widen and deepen operators so a larger student starts as the same function as a trained teacher. The goal is faster experimentation, not a controller that decides when to edit, and not a log of rejected edits.

Liu, Wu, and Wang, "Splitting Steepest Descent for Growing Neural Architectures," NeurIPS 2019. Grow by splitting neurons, with a criterion derived as a steepest descent in a Wasserstein space of neuron weights. The practical target they state is lightweight networks.

Wu, Liu, Stone, and Liu, "Firefly Neural Architecture Descent," NeurIPS 2020. A functional neighborhood of wider and deeper graphs, a Taylor step, and a greedy choice of which growth to apply. They report a continual-learning setting in which the grown nets are smaller and more accurate than the methods they compare. Firefly is the closest training-time growth method in this audit. It does not, in the abstract and introduction that were read, measure whether the growth policy itself improves, and it does not keep a ledger of rejected candidates.

Yang, Lin, Zhang, and Fan, "GROWN: GRow Only When Necessary for Continual Learning," arXiv:2110.00908 (2021). Start from a small seed, freeze old-task weights, reuse a subset, and expand with a learnable sparse growth only when an accuracy goal is missed. The "only when necessary" trigger is a threshold on task accuracy, not a shadow trial against a disjoint probe, and there is no random-edit control of the growth operator.

Jia and Zhou, "Self-Motivated Growing Neural Network for Adaptive Architecture via Local Structural Plasticity," arXiv:2512.12713. SMGrNN adds a structural plasticity module on a control policy. Local activity and weight-update statistics over a short window insert, explore, and prune, while backprop still trains the weights. The signals are the same family as our planned diagnostics. The task is policy distillation, and the paper does not describe a held-out probe gate or a Level-2 curve for the editor.

## Empirically validated self-improvement of code

Zhang, Hu, Lu, Lange, and Clune, "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents," arXiv:2505.22954, with a proceedings PDF under ICLR 2026 (https://proceedings.iclr.cc/paper_files/paper/2026/file/aa5f5e6eb6f613ec412f1d948dfa21a5-Paper-Conference.pdf). An archive of coding agents proposes edits to their own code. Each edit is kept or dropped by benchmark performance. Sakana's writeup is https://sakana.ai/dgm/. The gate is the idea we share: do not adopt a self-edit you have not measured. The substrate is repository code sampled by a foundation model. valmorph does not rewrite its own source, does not call a foundation model, and should not be described as a Gödel machine.

## Continual learning and plasticity

Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," PNAS 2017, DOI 10.1073/pnas.1611835114. Elastic weight consolidation slows learning on weights that a Fisher penalty marks as important to old tasks. This is a required baseline. It does not change architecture.

Dohare et al., "Loss of plasticity in deep continual learning," Nature 2024, https://www.nature.com/articles/s41586-024-07711-7. The author list on the page begins Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, and Sutton; the page display is abbreviated, so `paper/refs.bib` does not invent the missing names. The claim that was read: standard deep training on a non-stationary stream loses the ability to fit new data, which is not the same thing as forgetting old data. Continual backpropagation reinitializes a fraction of little-used units. Our plasticity metrics and the prune-and-regrow edit exist because of this distinction. We have not reproduced their ImageNet result, and we will not cite a number from the paper until that section is used as a sanity check.

## Novelty

The pieces above already exist: self-referential weights, fast weights, function-preserving morphisms, growth during continual learning, local structural plasticity, and empirical accept/reject of self-edits. None of the opened papers is "a small feedforward net whose discrete edits are all logged, including rejections, accepted only after a shadow trial on a probe split that is disjoint from test, compared with a random-edit controller, with a pre-registered test of whether the editor itself improves."

That combination is the hypothesis, not a finding. It may fail Level 1, Level 2, or both. The honest reading of the related work is that function-preserving growth and gated self-modification have both been shown in other settings, and that the open question for us is whether the gate and the editor help at a scale the M4 and a 20% slice of the 5080 can run repeatedly.

We will not claim to invent self-modification.

## Network morphism

Wei, Wang, Rui, and Chen, "Network Morphism," ICML 2016, PMLR 48, pages 564–572, arXiv:1603.01670. The PDF opened on 2026-10-27. A morphism here is a parameter map from a trained parent network to a larger child that preserves the function. They treat depth, width, kernel size, and subnet changes, including for convolutional nets, and they add a parametric activation so a nonlinear unit can be morphed. They describe inserting an identity layer, which they call IdMorph, as the operation in Chen, Goodfellow, and Shlens (Net2Net), and they say that identity layer is sparse. Our identity deepen is that insertion. We do not adopt their parametric activation.

## Continual-learning baselines opened on 2026-10-27

Lopez-Paz and Ranzato, "Gradient Episodic Memory for Continual Learning," NIPS 2017, arXiv:1706.08840. The model keeps an episodic memory of past tasks. A proposed update is kept when its inner product with each past task's memory gradient is non-negative. Otherwise the update is projected to the nearest vector that satisfies those inequalities, by a quadratic program whose dual has one variable per past task. Section 4.1 is the protocol note in ADR-013. Their published accuracies are not a sanity-check target yet.

Rolnick, Ahuja, Schwarz, Lillicrap, and Wayne, "Experience Replay for Continual Learning," NeurIPS 2019, arXiv:1811.11682. The PDF is a reinforcement-learning method, CLEAR: on-policy learning for plasticity, and off-policy replay plus behavioral cloning for stability. The domains they report are Atari and DMLab. Task boundaries are not given to the agent. A limited buffer drops stored events at random. This paper does not state a supervised classification replay formula. `train_with_rehearsal` stays our mix.

Farajtabar, Azizan, Mott, and Li, "Orthogonal Gradient Descent for Continual Learning," AISTATS 2020, PMLR 108, pages 3762–3773, arXiv:1910.07104. New-task gradients are projected onto the orthogonal complement of stored gradients of the network's outputs on previous data, so those outputs move as little as a local linear step allows. The method does not store the previous examples. Their appendix table on permuted MNIST is accuracy on task 1 after a second task, not the 20-task protocol below. Those accuracies are not our target.

Kornblith, Norouzi, Lee, and Hinton, "Similarity of Neural Network Representations Revisited," ICML 2019, PMLR 97, pages 3519–3529, arXiv:1905.00414. Linear CKA is the Hilbert-Schmidt independence criterion of the linear kernels, normalized by the two self-similarities. For centered example-by-feature matrices the numerator is the squared Frobenius norm of the cross-covariance. `linear_cka` implements that case. It is a diagnostic, not a result.

## Image protocols opened on 2026-10-27

Goodfellow, Mirza, Xiao, Courville, and Bengio, "An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks," arXiv:1312.6211. Section 4.1 is a pair of MNIST tasks that share the labels and differ by a pixel permutation. They call this input reformatting. It is not a 20-task sequence.

The 20-task permutation sequence we will use for H1 is section 4.1 of Lopez-Paz and Ranzato, written down in ADR-013. They cite Kirkpatrick et al. 2017 for the name "MNIST Permutations." Kirkpatrick's PDF was not re-opened for that section in this pass, so the task count is GEM's count, not a count read from Kirkpatrick.

Rebuffi, Kolesnikov, Sperl, and Lampert, "iCaRL: Incremental Classifier and Representation Learning," CVPR 2017, pages 2001–2010, arXiv:1611.07725. iCIFAR-100 trains all 100 classes in batches of 2, 5, 10, 20, or 50 classes, with a 32-layer ResNet, at most 2000 stored exemplars, and 70 epochs per batch. That schedule is not the same as GEM's incremental CIFAR-100, which uses 20 tasks of 5 classes and 2500 examples, one pass, and a smaller ResNet-18. Split CIFAR-100 is not frozen. ADR-013 says why.

## Still not opened

Kirkpatrick et al. 2017 was opened in the first related-work pass for the Fisher penalty. Its MNIST-permutation section was not the source of the task count above. Electricity's license on the OpenML page itself is still unread.
