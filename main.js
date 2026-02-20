class BaseModel {
    constructor() {
        this.params = {};
        this.isClassifier = false;
    }
    predict(x) {
        return 0;
    }
    fit(points) {}
    getMetrics(points) {
        if (this.isClassifier) return this.getClassificationMetrics(points);
        let sse = 0,
            sae = 0,
            sst = 0,
            n = points.length;
        const yVals = points.map(p => (250 - p.y) / 250);
        const yMean = yVals.reduce((a, b) => a + b, 0) / n;
        for (let p of points) {
            const xN = p.x / 600,
                yN = (250 - p.y) / 250;
            const pred = this.predict(xN);
            sse += Math.pow(pred - yN, 2);
            sae += Math.abs(pred - yN);
            sst += Math.pow(yN - yMean, 2);
        }
        return {
            m1: `MSE: ${(sse / n).toFixed(5)}`,
            m2: `MAE: ${(sae / n).toFixed(5)}`,
            m3: `R²: ${(sst === 0 ? 0 : 1 - (sse / sst)).toFixed(3)}`
        };
    }
    getClassificationMetrics(points) {
        let tp = 0,
            fp = 0,
            fn = 0,
            tn = 0;
        points.forEach(p => {
            const pred = this.predict(p.x / 600) > 0.5 ? 1 : 0;
            if (pred === 1 && p.label === 1) tp++;
            else if (pred === 1 && p.label === 0) fp++;
            else if (pred === 0 && p.label === 1) fn++;
            else if (pred === 0 && p.label === 0) tn++;
        });
        const acc = (tp + tn) / points.length;
        return {
            m1: `ACC: ${(acc * 100).toFixed(1)}%`,
            m2: `PREC: ${(tp / (tp + fp) || 0).toFixed(2)}`,
            m3: `REC: ${(tp / (tp + fn) || 0).toFixed(2)}`
        };
    }
}

class GradientModel extends BaseModel {
    constructor(type) {
        super();
        this.type = type;
        this.params = {
            a: 0.1,
            b: 1,
            c: 0.1,
            d: 0.5
        };
        this.lr = 0.1;
        this.iters = 4000;
    }
    predict(x) {
        const {
            a,
            b,
            c,
            d
        } = this.params;
        switch (this.type) {
            case 'Linear':
                return a * x + b;
            case 'Polynomial':
                return a * Math.pow(x - 0.5, 2) + b * x + c;
            case 'Exponential':
                return a * Math.exp(Math.min(b * x, 10)) + c;
            case 'Logarithmic':
                return a + b * Math.log(x + 0.05);
            case 'Periodic':
                return a * Math.sin(b * x + c) + d;
            case 'Logistic':
                return 1 / (1 + Math.exp(-(x - a) * b));
        }
    }
    fit(points) {
        let lr = this.lr,
            it = this.iters;
        if (this.type === 'Exponential') {
            this.params = {
                a: 0.1,
                b: 0.5,
                c: 0
            };
            lr = 0.01;
            it = 6000;
        }
        if (this.type === 'Logistic') {
            lr = 0.5;
            it = 5000;
        }
        if (this.type === 'Periodic') {
            let bestErr = Infinity,
                bestB = 5;
            for (let testB = 1; testB <= 25; testB += 0.5) {
                this.params = {
                    a: 0.3,
                    b: testB,
                    c: 0,
                    d: 0.5
                };
                let err = points.reduce((s, p) => s + Math.pow(this.predict(p.x / 600) - (250 - p.y) / 250, 2), 0);
                if (err < bestErr) {
                    bestErr = err;
                    bestB = testB;
                }
            }
            this.params = {
                a: 0.2,
                b: bestB,
                c: 0,
                d: 0.5
            };
            lr = 0.02;
            it = 12000;
        }
        for (let i = 0; i < it; i++) {
            let grads = {
                a: 0,
                b: 0,
                c: 0,
                d: 0
            };
            for (let p of points) {
                const x = p.x / 600,
                    y = (250 - p.y) / 250,
                    pred = this.predict(x),
                    err = pred - y;
                if (this.type === 'Linear') {
                    grads.a += err * x;
                    grads.b += err;
                } else if (this.type === 'Polynomial') {
                    grads.a += err * Math.pow(x - 0.5, 2);
                    grads.b += err * x;
                    grads.c += err;
                } else if (this.type === 'Exponential') {
                    const ex = Math.exp(Math.min(this.params.b * x, 10));
                    grads.a += err * ex;
                    grads.b += err * this.params.a * x * ex;
                    grads.c += err;
                } else if (this.type === 'Logarithmic') {
                    grads.a += err;
                    grads.b += err * Math.log(x + 0.05);
                } else if (this.type === 'Periodic') {
                    const arg = this.params.b * x + this.params.c,
                        cv = Math.cos(arg);
                    grads.a += err * Math.sin(arg);
                    grads.b += err * this.params.a * x * cv;
                    grads.c += err * this.params.a * cv;
                    grads.d += err;
                } else if (this.type === 'Logistic') {
                    const s = pred * (1 - pred);
                    grads.a += err * s * (-this.params.b);
                    grads.b += err * s * (x - this.params.a);
                }
            }
            for (let k in grads) this.params[k] -= (grads[k] / points.length) * lr;
        }
    }
}

class StepModel extends BaseModel {
    predict(x) {
        return x < this.params.a ? this.params.b : this.params.c;
    }
    fit(points) {
        let best = {
            a: 0.5,
            b: 0,
            c: 0,
            err: Infinity
        };
        for (let tA = 0.1; tA < 0.9; tA += 0.02) {
            const l = points.filter(p => (p.x / 600) < tA),
                r = points.filter(p => (p.x / 600) >= tA);
            if (!l.length || !r.length) continue;
            const b = l.reduce((s, p) => s + (250 - p.y) / 250, 0) / l.length,
                c = r.reduce((s, p) => s + (250 - p.y) / 250, 0) / r.length;
            const err = points.reduce((s, p) => s + Math.pow(((p.x / 600) < tA ? b : c) - (250 - p.y) / 250, 2), 0);
            if (err < best.err) best = {
                a: tA,
                b,
                c,
                err
            };
        }
        this.params = best;
    }
}

class NaiveBayesModel extends BaseModel {
    constructor() {
        super();
        this.isClassifier = true;
        this.stats = {};
    }
    fit(points) {
        [0, 1].forEach(c => {
            const vals = points.filter(p => p.label === c).map(p => p.x / 600);
            if (vals.length < 2) {
                this.stats[c] = {
                    m: c === 1 ? 0.7 : 0.3,
                    s: 0.15,
                    p: 0.5
                };
                return;
            }
            const m = vals.reduce((a, b) => a + b, 0) / vals.length;
            const v = vals.reduce((a, b) => a + Math.pow(b - m, 2), 0) / vals.length;
            this.stats[c] = {
                m,
                s: Math.sqrt(v + 0.002),
                p: vals.length / points.length
            };
        });
    }
    predict(x) {
        const probs = [0, 1].map(c => {
            const {
                m,
                s,
                p
            } = this.stats[c];
            if (p === 0) return 0.0001;
            const exponent = Math.exp(-Math.pow(x - m, 2) / (2 * s * s));
            return p * (1 / (Math.sqrt(2 * Math.PI) * s)) * exponent;
        });
        const sum = probs[0] + probs[1];
        return sum === 0 ? 0.5 : probs[1] / sum;
    }
}

class KNNModel extends BaseModel {
    constructor(k = 5) {
        super();
        this.isClassifier = true;
        this.k = k;
        this.data = [];
    }
    fit(points) {
        this.data = points;
    }
    predict(x) {
        if (!this.data.length) return 0;
        const dists = this.data.map(p => ({
            d: Math.abs(x - (p.x / 600)),
            label: p.label
        })).sort((a, b) => a.d - b.d);
        const votes = dists.slice(0, this.k).reduce((acc, n) => {
            acc[n.label]++;
            return acc;
        }, {
            0: 0,
            1: 0
        });
        return votes[1] / this.k;
    }
}

class SVMModel extends BaseModel {
    constructor() {
        super();
        this.isClassifier = true;
        this.landmarks = [];
        this.weights = [];
        this.b = 0;
    }
    kernel(x1, x2) {
        return Math.exp(-15.0 * Math.pow(x1 - x2, 2));
    }
    fit(points) {
        const xVals = points.map(p => p.x / 600),
            yLabels = points.map(p => p.label === 1 ? 1 : -1);
        this.landmarks = Array.from({
            length: 15
        }, (_, i) => i / 14);
        this.weights = new Array(this.landmarks.length).fill(0);
        this.b = 0;
        let lr = 0.1;
        const lambda = 0.001,
            iters = 10000;
        for (let i = 0; i < iters; i++) {
            const idx = Math.floor(Math.random() * points.length),
                x = xVals[idx],
                y = yLabels[idx];
            let pred = this.b;
            for (let j = 0; j < this.landmarks.length; j++) pred += this.weights[j] * this.kernel(this.landmarks[j], x);
            if (y * pred < 1) {
                const grad = y * (1 - y * pred);
                for (let j = 0; j < this.landmarks.length; j++) this.weights[j] += lr * (grad * this.kernel(this.landmarks[j], x) - 2 * lambda * this.weights[j]);
                this.b += lr * grad;
            } else {
                for (let j = 0; j < this.landmarks.length; j++) this.weights[j] -= lr * (2 * lambda * this.weights[j]);
            }
            lr *= 0.9999;
        }
    }
    predict(x) {
        let score = this.b;
        for (let j = 0; j < this.landmarks.length; j++) score += this.weights[j] * this.kernel(this.landmarks[j], x);
        return 1 / (1 + Math.exp(-score * 2));
    }
}

class DecisionTree {
    constructor(depth = 4) {
        this.depth = depth;
        this.root = null;
    }
    fit(points) {
        this.root = this.build(points, 0);
    }
    build(points, d) {
        if (d >= this.depth || points.length < 2 || points.every(p => p.label === points[0].label)) {
            const sum = points.reduce((s, p) => s + p.label, 0);
            return { leaf: true, val: sum / points.length || 0 };
        }
        let best = { split: 0, err: Infinity };
        for (let i = 0; i < points.length; i++) {
            const split = points[i].x / 600;
            const l = points.filter(p => p.x / 600 < split);
            const r = points.filter(p => p.x / 600 >= split);
            if (!l.length || !r.length) continue;
            const err = this.calcErr(l) + this.calcErr(r);
            if (err < best.err) best = { split, l, r };
        }
        return {
            leaf: false,
            split: best.split,
            left: this.build(best.l, d + 1),
            right: this.build(best.r, d + 1)
        };
    }
    calcErr(pts) {
        const m = pts.reduce((a, b) => a + b.label, 0) / pts.length;
        return pts.reduce((a, b) => a + Math.pow(b.label - m, 2), 0);
    }
    predict(x) {
        let n = this.root;
        while (n && !n.leaf) n = x < n.split ? n.left : n.right;
        return n ? n.val : 0.5;
    }
}

class TreeRegressorModel extends BaseModel {
    constructor(depth = 4) {
        super();
        this.depth = depth;
        this.root = null;
    }
    fit(points) {
        if (!points || points.length === 0) return;
        const data = points.map(p => ({ x: p.x / 600, y: (250 - p.y) / 250 }));
        this.root = this.build(data, 0);
    }
    build(pts, d) {
        const meanY = pts.reduce((s, p) => s + p.y, 0) / pts.length;
        if (d >= this.depth || pts.length <= 2) {
            return { leaf: true, val: meanY || 0 };
        }
        let best = { split: null, err: Infinity, l: [], r: [] };
        const sorted = [...pts].sort((a, b) => a.x - b.x);
        
        for (let i = 0; i < sorted.length - 1; i++) {
            const split = (sorted[i].x + sorted[i+1].x) / 2;
            const l = sorted.slice(0, i + 1);
            const r = sorted.slice(i + 1);
            const err = this.calcVar(l) + this.calcVar(r);
            if (err < best.err) best = { split, err, l, r };
        }
        
        if (best.split === null) return { leaf: true, val: meanY };

        return {
            leaf: false,
            split: best.split,
            left: this.build(best.l, d + 1),
            right: this.build(best.r, d + 1)
        };
    }
    calcVar(pts) {
        if (pts.length === 0) return 0;
        const m = pts.reduce((s, p) => s + p.y, 0) / pts.length;
        return pts.reduce((s, p) => s + Math.pow(p.y - m, 2), 0);
    }
    predict(x) {
        if (!this.root) return 0.5;
        let n = this.root;
        while (!n.leaf) {
            n = x < n.split ? n.left : n.right;
        }
        return n.val;
    }
}

class ForestModel extends BaseModel {
    constructor(count = 10) {
        super();
        this.isClassifier = true;
        this.count = count;
        this.depth = 4;
        this.trees = [];
    }
    fit(points) {
        this.trees = Array.from({ length: this.count }, () => new DecisionTree(this.depth));
        this.trees.forEach(t => {
            const sample = Array.from({ length: points.length }, () => points[Math.floor(Math.random() * points.length)]);
            t.fit(sample);
        });
    }
    predict(x) {
        if (!this.trees.length) return 0.5;
        return this.trees.reduce((a, b) => a + b.predict(x), 0) / this.trees.length;
    }
}

class ModelViz {
    constructor(type, formula, container, modelClass, isClassifier = false) {
        this.type = type;
        this.isClassifier = isClassifier;
        const clone = document.getElementById('model-template').content.cloneNode(true);
        this.el = clone.querySelector('.model-card');
        this.canvas = clone.querySelector('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.labels = [this.el.querySelector('.mse-value'), this.el.querySelector('.mae-value'), this.el.querySelector('.r2-value')];
        this.inputs = {
            spread: this.el.querySelector('.spread-input'),
            outlier: this.el.querySelector('.outlier-input')
        };
        this.el.querySelector('.model-title').textContent = type;
        this.el.querySelector('.model-formula').textContent = formula;
        container.appendChild(this.el);
        this.model = modelClass;
        this.points = [];
        this.el.querySelector('.reroll-btn').onclick = () => this.generate();
        Object.values(this.inputs).forEach(i => i.oninput = () => this.generate());
        this.generate();

        if (type === 'Decision Forest' || type === 'Decision Tree Regressor') {
            const controls = this.el.querySelector('.local-controls');
            const depthContainer = document.createElement('div');
            depthContainer.className = 'control-group';
            depthContainer.innerHTML = `
                <label>Tree Depth: <span class="depth-val">4</span></label>
                <input type="range" class="depth-input" min="1" max="10" value="4">
            `;
            controls.appendChild(depthContainer);
            this.inputs.depth = depthContainer.querySelector('.depth-input');
            this.depthDisplay = depthContainer.querySelector('.depth-val');
            this.inputs.depth.oninput = (e) => {
                this.depthDisplay.textContent = e.target.value;
                this.model.depth = parseInt(e.target.value);
                this.generate();
            };
        }
    }
    generate() {
        const spr = this.inputs.spread.value / 100,
            out = this.inputs.outlier.value / 100;
        this.points = [];
        const isComplex = Math.random() > 0.4;

        for (let i = 0; i < 100; i++) {
            let xN = Math.random(), label, yN;
            if (this.isClassifier) {
                label = isComplex ? ((xN > 0.4 && xN < 0.6) ? 1 : 0) : (xN > 0.5 ? 1 : 0);
                xN += (Math.random() - 0.5) * spr * 0.5;
                if (Math.random() < out) label = 1 - label;
                yN = 0.3 + Math.random() * 0.4;
            } else {
                switch (this.type) {
                    case 'Linear': yN = 0.5 * xN + 0.2; break;
                    case 'Polynomial': yN = 2.5 * Math.pow(xN - 0.5, 2) + 0.2; break;
                    case 'Exponential': yN = 0.15 * Math.exp(1.8 * xN) + 0.1; break;
                    case 'Logarithmic': yN = 0.5 + 0.2 * Math.log(xN + 0.01); break;
                    case 'Periodic': yN = 0.2 * Math.sin(10 * xN) + 0.5; break;
                    case 'Step': yN = xN < 0.5 ? 0.2 : 0.8; break;
                    case 'Logistic': yN = 1 / (1 + Math.exp(-(xN - 0.5) * 10)); break;
                    case 'Decision Tree Regressor': yN = 0.4 * Math.sin(5 * xN) + 0.5; break; 
                    default: yN = 0.5;
                }
                if (Math.random() < out) yN = Math.random();
                else yN += (Math.random() - 0.5) * spr;
            }
            this.points.push({
                x: Math.max(0, Math.min(1, xN)) * 600,
                y: 250 - (Math.max(0, Math.min(1, yN)) * 250),
                label
            });
        }
        this.model.fit(this.points);
        const m = this.model.getMetrics(this.points);
        [m.m1, m.m2, m.m3].forEach((v, i) => this.labels[i].textContent = v);
        this.draw();
    }
    draw() {
        this.ctx.clearRect(0, 0, 600, 250);
        if (this.isClassifier) {
            let lastProb = this.model.predict(0);
            for (let x = 0; x < 600; x += 2) {
                const prob = this.model.predict(x / 600);
                this.ctx.fillStyle = `rgba(${255 * (1 - prob)}, ${123 * prob + 166 * (1 - prob)}, ${114 * (1 - prob) + 255 * prob}, 0.25)`;
                this.ctx.fillRect(x, 0, 2, 250);

                if ((lastProb <= 0.5 && prob > 0.5) || (lastProb >= 0.5 && prob < 0.5)) {
                    this.ctx.strokeStyle = 'rgba(255,255,255,0.8)';
                    this.ctx.lineWidth = 2;
                    this.ctx.beginPath();
                    this.ctx.moveTo(x, 0);
                    this.ctx.lineTo(x, 250);
                    this.ctx.stroke();
                }
                lastProb = prob;
            }
        }
        this.points.forEach(p => {
            this.ctx.shadowBlur = 4;
            this.ctx.shadowColor = 'rgba(0,0,0,0.5)';
            this.ctx.fillStyle = this.isClassifier ? (p.label === 1 ? '#58a6ff' : '#ff7b72') : '#58a6ff';
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
            this.ctx.strokeStyle = '#fff';
            this.ctx.lineWidth = 0.5;
            this.ctx.stroke();
        });
        if (!this.isClassifier) {
            this.ctx.strokeStyle = '#ff7b72';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            for (let x = 0; x <= 600; x += 2) this.ctx.lineTo(x, 250 - (this.model.predict(x / 600) * 250));
            this.ctx.stroke();
        }
    }
}

const container = document.getElementById('models-container');
const reg = (t, f) => new ModelViz(t, f, container, new GradientModel(t));
reg('Linear', 'y = ax + b');
reg('Polynomial', 'y = a(x-0.5)² + bx + c');
reg('Exponential', 'y = ae^(bx) + c');
reg('Logarithmic', 'y = a + b * ln(x)');
reg('Periodic', 'y = a * sin(bx + c) + d');
reg('Logistic', 'y = 1 / (1 + e^-z)');
new ModelViz('Step', 'y = (x < a) ? b : c', container, new StepModel());
new ModelViz('Naive Bayes', 'P(C|x) ∝ P(x|C)P(C)', container, new NaiveBayesModel(), true);
new ModelViz('KNN', 'k = 5, Neighbors', container, new KNNModel(5), true);
new ModelViz('SVM', 'RBF Kernel, Soft Margin', container, new SVMModel(), true);
const forest = new ForestModel(12);
new ModelViz('Decision Forest', 'Bagging & Ensemble Splits', container, forest, true);
new ModelViz('Decision Tree Regressor', 'Recursive Mean Splitting', container, new TreeRegressorModel(4));