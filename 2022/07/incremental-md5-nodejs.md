---
title: Incremental MD5 Hash with Node-JS
subtitle: Dealing with large buffers of data efficiently 
slug: 2022-07-node-js-md5-hash-incremental
tags: javascript, nodejs, coding, algorithms
cover: https://raw.githubusercontent.com/shailesh1729/hashnode-articles/main/2022/07/images/md5-hash.png
domain: shailesh.hashnode.dev
---

A cryptographic hash is a function which can map data of arbitrary size
to fixed size values. 
A good hash function satisfies two basic properties:

1. It should be very fast to compute. 
1. It should minimize duplication of output values (collisions).
   In other words, it should be unlikely that two different
   input data lead to the same output value.

`md5` is a popular algorithm for computing hash of data.
NODE-JS has complete support for the computation for md5 hashes
on arbitrary strings or buffer. 

When working with large data files, it is useful
to compute the hash incrementally.
Here is a particular scenario I faced.
In my Express JS backend, I was computing the MD5
hash of an uploaded file [multipart file upload request].
For large files, the simple `md5` function provided
by the [md5](https://github.com/pvorb/node-md5)
library was crashing on a low RAM EC2 instance.
I then reverted to an incremental hash computation
using the core functionality provided by the
`crypto` module.

The incremental algorithm looks like following:

1. Split the data into chunks.
1. Initiate the md5 hash for an empty data buffer.
1. For each data chunk, update the hash.
1. Once all the data chunks have been processed, then return the final hash value.


The `crypto` library provides complete support for computing md5 hashes
incrementally. Let us see how to do this for a large file.


Load the relevant libraries
```javascript
const fs = require('fs');
const crypto = require('crypto');
```

Read a large file from the disk (please provide the file path)
```javascript
const data = fs.readFileSync(filepath);
```

Decide the size of each data chunk to be processed

```javascript
const chunk_size = 4096;
```

A variable to hold data chunks:
```javascript
let chunk = null;
```

The MD5 hash object
```javascript
const hasher = crypto.createHash('md5');
```

Total data length
```javascript
const n = data.length;
```

Initial position in the data buffer
```javascript
let i = 0; 
```

Extract the chunks from data iteratively
and update the hash using the current chunk:
```javascript
// We shall process till we have exhausted all data
while (i < n){
    // Length of the remaining chunk
    const chunk_len = Math.min(n-i, chunk_size);
    // Data of the chunk
    chunk = data.slice(i, i + chunk_len);
    // Update the MD5 hash using the chunk data
    hasher.update(chunk);
    // Move on to the next chunk
    i += chunk_size;
}
```

We can now get the hash value from the MD5 hash object

```javascript
const hash_value = hasher.digest('hex');
```


The function below puts together all the
steps of splitting data into chunks and
then incrementally computing the hash.


```javascript
const compute_md5_hash = (buffer) => {
    let i = 0;
    let k = 0;
    let n = buffer.length;
    let chunk_size = 4096;
    let chunk = null;
    const hash = crypto.createHash('md5');
    while (i < n){
        k = k + 1;
        const chunk_len = Math.min(n-i, chunk_size);
        chunk = buffer.slice(i, i + chunk_len);
        i += chunk_size;
        hash.update(chunk);
    }
    return hash.digest('hex');
}
```
