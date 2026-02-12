# ============================================================
# Fast dependency-free base R replication of the same
# pure-python micrograd+GPT algorithm.
# 
# https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
# ============================================================

set.seed(42)

# -------------------------
# Data download + dataset
# -------------------------
if (!file.exists("input.txt")) {
  names_url <- "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
  download.file(names_url, "input.txt", quiet = TRUE)
}

docs <- readLines("input.txt", warn = FALSE)
docs <- trimws(docs)
docs <- docs[nzchar(docs)]
docs <- sample(docs, length(docs))
cat(sprintf("num docs: %d\n", length(docs)))

# -------------------------
# Tokenizer (chars)
# -------------------------
all_chars <- unlist(strsplit(paste(docs, collapse = ""), split = "", fixed = TRUE), use.names = FALSE)
uchars <- sort(unique(all_chars))
vocab_size <- length(uchars) + 1
BOS_ID <- length(uchars) + 1
cat(sprintf("vocab size: %d\n", vocab_size))

# Fast char->id mapping via named integer vector (1-based IDs)
char_to_id <- setNames(seq_along(uchars), uchars)

# Pre-tokenize docs once: tokens = [BOS] + chars + [BOS]
tokenized_docs <- vector("list", length(docs))
for (di in seq_along(docs)) {
  ch <- unlist(strsplit(docs[[di]], "", fixed = TRUE), use.names = FALSE)
  ids <- unname(char_to_id[ch])
  tokenized_docs[[di]] <- c(BOS_ID, ids, BOS_ID)
}

# -------------------------
# Autograd Value (micrograd)
# -------------------------
Value <- function(data, children = NULL, local_grads = NULL) {
  e <- new.env(parent = emptyenv())
  e$data <- as.numeric(data)
  e$grad <- 0.0
  e$children <- children  # list of envs or NULL
  e$local_grads <- local_grads # numeric or NULL
  e
}

as_value <- function(x) {
  if (is.environment(x) && !is.null(x$data)) return(x)
  Value(x)
}

v_add <- function(a, b) {
  a <- as_value(a); b <- as_value(b)
  Value(a$data + b$data, list(a, b), c(1.0, 1.0))
}

v_mul <- function(a, b) {
  a <- as_value(a); b <- as_value(b)
  Value(a$data * b$data, list(a, b), c(b$data, a$data))
}

v_neg <- function(a) v_mul(a, -1.0)
v_sub <- function(a, b) v_add(a, v_neg(b))

v_pow <- function(a, p) {
  a <- as_value(a)
  Value(a$data^p, list(a), c(p * (a$data^(p - 1.0))))
}

v_log <- function(a) {
  a <- as_value(a)
  Value(log(a$data), list(a), c(1.0 / a$data))
}

v_exp <- function(a) {
  a <- as_value(a)
  ex <- exp(a$data)
  Value(ex, list(a), c(ex))
}

v_relu <- function(a) {
  a <- as_value(a)
  Value(if (a$data > 0) a$data else 0.0, list(a), c(if (a$data > 0) 1.0 else 0.0))
}

# Backprop with topo sort (same idea, fewer closures)
backward <- function(root) {
  topo <- vector("list", 0)
  visited <- new.env(parent = emptyenv())

  stack <- list(root)
  post <- list()
  while (length(stack) > 0) {
    v <- stack[[length(stack)]]
    stack[[length(stack)]] <- NULL
    key <- sprintf("%p", v)
    if (is.null(visited[[key]])) {
      visited[[key]] <- TRUE
      post[[length(post) + 1]] <- v
      ch <- v$children
      if (!is.null(ch)) {
        for (i in seq_along(ch)) stack[[length(stack) + 1]] <- ch[[i]]
      }
    }
  }

  # reverse postorder: ensure children before parents
  # we built "post" in discovery order; do a second pass to topo-sort properly
  # simplest: do recursive topo like earlier, but keep it fast-ish:
  topo <- vector("list", 0)
  visited2 <- new.env(parent = emptyenv())
  build <- function(v) {
    key <- sprintf("%p", v)
    if (is.null(visited2[[key]])) {
      visited2[[key]] <- TRUE
      ch <- v$children
      if (!is.null(ch)) for (i in seq_along(ch)) build(ch[[i]])
      topo[[length(topo) + 1]] <<- v
    }
  }
  build(root)

  root$grad <- 1.0
  for (i in length(topo):1) {
    v <- topo[[i]]
    ch <- v$children
    if (!is.null(ch)) {
      lg <- v$local_grads
      for (j in seq_along(ch)) {
        child <- ch[[j]]
        child$grad <- child$grad + lg[[j]] * v$grad
      }
    }
  }
}

# -------------------------
# Parameters (state_dict)
# -------------------------
n_embd <- 16
n_head <- 4
n_layer <- 1
block_size <- 16
head_dim <- n_embd / n_head

matrix <- function(nout, nin, std = 0.08) {
  mat <- vector("list", nout)
  for (o in 1:nout) {
    row <- vector("list", nin)
    for (i in 1:nin) row[[i]] <- Value(rnorm(1, 0, std))
    mat[[o]] <- row
  }
  mat
}

state_dict <- list(
  wte = matrix(vocab_size, n_embd),
  wpe = matrix(block_size, n_embd),
  lm_head = matrix(vocab_size, n_embd)
)

for (li in 0:(n_layer - 1)) {
  state_dict[[sprintf("layer%d.attn_wq", li)]] <- matrix(n_embd, n_embd)
  state_dict[[sprintf("layer%d.attn_wk", li)]] <- matrix(n_embd, n_embd)
  state_dict[[sprintf("layer%d.attn_wv", li)]] <- matrix(n_embd, n_embd)
  state_dict[[sprintf("layer%d.attn_wo", li)]] <- matrix(n_embd, n_embd)
  state_dict[[sprintf("layer%d.mlp_fc1", li)]] <- matrix(4 * n_embd, n_embd)
  state_dict[[sprintf("layer%d.mlp_fc2", li)]] <- matrix(n_embd, 4 * n_embd)
}

# Flatten params for Adam (same as before)
params <- vector("list", 0)
for (mat in state_dict) for (row in mat) for (p in row) params[[length(params) + 1]] <- p
cat(sprintf("num params: %d\n", length(params)))

# -------------------------
# Faster core ops (same math)
# -------------------------

# sum_i (w_i * x_i) without building intermediate lists
dot_wx <- function(w_row, x) {
  acc <- Value(0.0)
  for (i in seq_along(x)) {
    acc <- v_add(acc, v_mul(w_row[[i]], x[[i]]))
  }
  acc
}

linear <- function(x, w) {
  out <- vector("list", length(w))
  for (o in seq_along(w)) {
    out[[o]] <- dot_wx(w[[o]], x)
  }
  out
}

softmax <- function(logits) {
  # stable softmax: subtract max
  max_val <- -Inf
  for (i in seq_along(logits)) if (logits[[i]]$data > max_val) max_val <- logits[[i]]$data

  exps <- vector("list", length(logits))
  total <- Value(0.0)
  for (i in seq_along(logits)) {
    exps[[i]] <- v_exp(v_sub(logits[[i]], max_val))
    total <- v_add(total, exps[[i]])
  }
  probs <- vector("list", length(logits))
  for (i in seq_along(logits)) probs[[i]] <- v_mul(exps[[i]], v_pow(total, -1.0))
  probs
}

rmsnorm <- function(x) {
  # ms = mean(x^2); scale=(ms+1e-5)^-0.5; return x*scale
  acc <- Value(0.0)
  for (i in seq_along(x)) acc <- v_add(acc, v_mul(x[[i]], x[[i]]))
  ms <- v_mul(acc, 1.0 / length(x))
  scale <- v_pow(v_add(ms, 1e-5), -0.5)

  y <- vector("list", length(x))
  for (i in seq_along(x)) y[[i]] <- v_mul(x[[i]], scale)
  y
}

# -------------------------
# GPT forward (same algorithm, fewer allocations)
# -------------------------
gpt <- function(token_id, pos_id, keys, values) {
  tok_emb <- state_dict$wte[[token_id]]
  pos_emb <- state_dict$wpe[[pos_id]]

  x <- vector("list", n_embd)
  for (i in 1:n_embd) x[[i]] <- v_add(tok_emb[[i]], pos_emb[[i]])
  x <- rmsnorm(x)

  for (li in 0:(n_layer - 1)) {
    # attention block
    x_res <- x
    x <- rmsnorm(x)

    wq <- state_dict[[sprintf("layer%d.attn_wq", li)]]
    wk <- state_dict[[sprintf("layer%d.attn_wk", li)]]
    wv <- state_dict[[sprintf("layer%d.attn_wv", li)]]
    wo <- state_dict[[sprintf("layer%d.attn_wo", li)]]

    q <- linear(x, wq)
    k <- linear(x, wk)
    v <- linear(x, wv)

    keys[[li + 1]][[length(keys[[li + 1]]) + 1]] <- k
    values[[li + 1]][[length(values[[li + 1]]) + 1]] <- v

    x_attn <- vector("list", 0)
    Tlen <- length(keys[[li + 1]])

    for (h in 0:(n_head - 1)) {
      hs <- h * head_dim
      idx <- (hs + 1):(hs + head_dim)
      q_h <- q[idx]

      # logits over time
      attn_logits <- vector("list", Tlen)
      inv_sqrt_hd <- 1.0 / sqrt(head_dim)

      for (t in 1:Tlen) {
        k_t <- keys[[li + 1]][[t]][idx]
        acc <- Value(0.0)
        for (j in 1:head_dim) acc <- v_add(acc, v_mul(q_h[[j]], k_t[[j]]))
        attn_logits[[t]] <- v_mul(acc, inv_sqrt_hd)
      }

      attn_w <- softmax(attn_logits)

      # weighted sum of v over time
      head_out <- vector("list", head_dim)
      for (j in 1:head_dim) head_out[[j]] <- Value(0.0)

      for (t in 1:Tlen) {
        v_t <- values[[li + 1]][[t]][idx]
        wt <- attn_w[[t]]
        for (j in 1:head_dim) {
          head_out[[j]] <- v_add(head_out[[j]], v_mul(wt, v_t[[j]]))
        }
      }

      x_attn <- c(x_attn, head_out)
    }

    x <- linear(x_attn, wo)
    for (i in 1:n_embd) x[[i]] <- v_add(x[[i]], x_res[[i]])

    # MLP block
    x_res <- x
    x <- rmsnorm(x)

    fc1 <- state_dict[[sprintf("layer%d.mlp_fc1", li)]]
    fc2 <- state_dict[[sprintf("layer%d.mlp_fc2", li)]]

    x <- linear(x, fc1)
    for (i in seq_along(x)) x[[i]] <- v_relu(x[[i]])
    x <- linear(x, fc2)

    for (i in 1:n_embd) x[[i]] <- v_add(x[[i]], x_res[[i]])
  }

  linear(x, state_dict$lm_head)
}

# -------------------------
# Adam (same)
# -------------------------
learning_rate <- 0.01
beta1 <- 0.85
beta2 <- 0.99
eps_adam <- 1e-8

m <- rep(0.0, length(params))
vv <- rep(0.0, length(params))

# -------------------------
# Training loop (same algorithm)
# -------------------------
num_steps <- 1000

for (step in 0:(num_steps - 1)) {
  tokens <- tokenized_docs[[ (step %% length(tokenized_docs)) + 1 ]]
  n <- min(block_size, length(tokens) - 1)

  keys <- replicate(n_layer, list(), simplify = FALSE)
  values <- replicate(n_layer, list(), simplify = FALSE)

  # accumulate loss
  loss_acc <- Value(0.0)
  for (pos_id in 1:n) {
    token_id <- tokens[[pos_id]]
    target_id <- tokens[[pos_id + 1]]

    logits <- gpt(token_id, pos_id, keys, values)
    probs <- softmax(logits)

    # loss_t = -log(probs[target])
    loss_t <- v_neg(v_log(probs[[target_id]]))
    loss_acc <- v_add(loss_acc, loss_t)
  }

  loss <- v_mul(loss_acc, 1.0 / n)

  backward(loss)

  lr_t <- learning_rate * (1.0 - step / num_steps)
  for (i in seq_along(params)) {
    p <- params[[i]]
    g <- p$grad

    m[i]  <- beta1 * m[i]  + (1.0 - beta1) * g
    vv[i] <- beta2 * vv[i] + (1.0 - beta2) * (g * g)

    m_hat <- m[i] / (1.0 - beta1^(step + 1))
    v_hat <- vv[i] / (1.0 - beta2^(step + 1))

    p$data <- p$data - lr_t * m_hat / (sqrt(v_hat) + eps_adam)
    p$grad <- 0.0
  }

  cat(sprintf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss$data))
}

# -------------------------
# Inference (sampling, same)
# -------------------------
temperature <- 0.5
id_to_char <- uchars

cat("\n--- inference (new, hallucinated names) ---\n")
for (sample_idx in 1:20) {
  keys <- replicate(n_layer, list(), simplify = FALSE)
  values <- replicate(n_layer, list(), simplify = FALSE)
  token_id <- BOS_ID
  out <- character(0)

  for (pos_id in 1:block_size) {
    logits <- gpt(token_id, pos_id, keys, values)

    # logits / temperature
    for (i in seq_along(logits)) logits[[i]] <- v_mul(logits[[i]], 1.0 / temperature)

    probs <- softmax(logits)
    weights <- vapply(probs, function(p) p$data, numeric(1))

    token_id <- sample.int(vocab_size, size = 1, prob = weights)
    if (token_id == BOS_ID) break
    out <- c(out, id_to_char[[token_id]])
  }

  cat(sprintf("sample %2d: %s\n", sample_idx, paste(out, collapse = "")))
}
