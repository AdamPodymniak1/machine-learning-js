class BaseModel {
    constructor() {
        this.params = {};
        this.isClassifier = false;
    }
    predict(x) { return 0; }
    fit(points) {}
    getMetrics(points) {
        if (this.isClassifier) return this.getClassificationMetrics(points);
        let sse = 0, sae = 0, sst = 0;
        const n = points.length;
        const yVals = points.map(p => (250 - p.y) / 250);
        const yMean = yVals.reduce((a, b) => a + b, 0) / n;
        for (let p of points) {
            const xN = p.x / 600, yN = (250 - p.y) / 250;
            const pred = this.predict(xN);
            sse += Math.pow(pred - yN, 2);
            sae += Math.abs(pred - yN);
            sst += Math.pow(yN - yMean, 2);
        }
        return {
            m1: `MSE: ${(sse / n).toFixed(5)}`,
            m2: `MAE: ${(sae / n).toFixed(5)}`,
            m3: `R2: ${(sst === 0 ? 0 : 1 - (sse / sst)).toFixed(3)}`
        };
    }
    getClassificationMetrics(points) {
        let tp = 0, fp = 0, fn = 0, tn = 0;
        points.forEach(p => {
            const pred = this.predict(p.x / 600) > 0.5 ? 1 : 0;
            if (pred === 1 && p.label === 1) tp++;
            else if (pred === 1 && p.label === 0) fp++;
            else if (pred === 0 && p.label === 1) fn++;
            else if (pred === 0 && p.label === 0) tn++;
        });
        const acc = (tp + tn) / points.length;
        const prec = tp / (tp + fp) || 0;
        const rec = tp / (tp + fn) || 0;
        return {
            m1: `ACC: ${(acc * 100).toFixed(1)}%`,
            m2: `PREC: ${prec.toFixed(2)}`,
            m3: `REC: ${rec.toFixed(2)}`
        };
    }
}

class GradientModel extends BaseModel {
    constructor(type) {
        super();
        this.type = type;
        this.params = { a: 0.1, b: 1, c: 0.1, d: 0.5 };
        this.lr = 0.1;
        this.iters = 4000;
    }
    predict(x) {
        const { a, b, c, d } = this.params;
        switch (this.type) {
            case 'Linear': return a * x + b;
            case 'Polynomial': return a * Math.pow(x - 0.5, 2) + b * x + c;
            case 'Exponential': return a * Math.exp(Math.min(b * x, 10)) + c;
            case 'Logarithmic': return a + b * Math.log(x + 0.05);
            case 'Periodic': return a * Math.sin(b * x + c) + d;
            case 'Logistic': return 1 / (1 + Math.exp(-(x - a) * b));
        }
    }
    fit(points) {
        let lr = this.lr, it = this.iters;
        if (this.type === 'Exponential') { this.params = { a: 0.1, b: 0.5, c: 0 }; lr = 0.01; it = 6000; }
        if (this.type === 'Logistic') { lr = 0.5; it = 5000; }
        if (this.type === 'Periodic') { this.findBestPeriodicStart(points); lr = 0.02; it = 10000; }
        for (let i = 0; i < it; i++) {
            let grads = { a: 0, b: 0, c: 0, d: 0 };
            for (let p of points) {
                const x = p.x / 600, y = (250 - p.y) / 250;
                const pred = this.predict(x), err = pred - y;
                if (this.type === 'Linear') { grads.a += err * x; grads.b += err; }
                else if (this.type === 'Polynomial') { grads.a += err * Math.pow(x - 0.5, 2); grads.b += err * x; grads.c += err; }
                else if (this.type === 'Exponential') {
                    const ex = Math.exp(Math.min(this.params.b * x, 10));
                    grads.a += err * ex; grads.b += err * this.params.a * x * ex; grads.c += err;
                } else if (this.type === 'Logarithmic') { grads.a += err; grads.b += err * Math.log(x + 0.05); }
                else if (this.type === 'Periodic') {
                    const cv = Math.cos(this.params.b * x + this.params.c);
                    grads.a += err * Math.sin(this.params.b * x + this.params.c);
                    grads.b += err * this.params.a * x * cv; grads.c += err * this.params.a * cv; grads.d += err;
                } else if (this.type === 'Logistic') {
                    const s = pred * (1 - pred);
                    grads.a += err * s * (-this.params.b); grads.b += err * s * (x - this.params.a);
                }
            }
            for (let k in grads) {
                let delta = (grads[k] / points.length) * lr;
                this.params[k] -= Math.max(-0.5, Math.min(0.5, delta)); 
            }
        }
    }
    findBestPeriodicStart(points) {
        let bestErr = Infinity, bestB = 5;
        for (let testB of [5, 10, 15, 20]) {
            this.params = { a: 0.2, b: testB, c: 0, d: 0.5 };
            let err = points.reduce((s, p) => s + Math.pow(this.predict(p.x/600) - (250-p.y)/250, 2), 0);
            if (err < bestErr) { bestErr = err; bestB = testB; }
        }
        this.params = { a: 0.2, b: bestB, c: 0, d: 0.5 };
    }
}

class StepModel extends BaseModel {
    predict(x) { return x < this.params.a ? this.params.b : this.params.c; }
    fit(points) {
        let best = { a: 0.5, b: 0, c: 0, err: Infinity };
        for (let tA = 0.1; tA < 0.9; tA += 0.02) {
            const l = points.filter(p => (p.x / 600) < tA), r = points.filter(p => (p.x / 600) >= tA);
            if (!l.length || !r.length) continue;
            const b = l.reduce((s, p) => s + (250 - p.y) / 250, 0) / l.length;
            const c = r.reduce((s, p) => s + (250 - p.y) / 250, 0) / r.length;
            const err = points.reduce((s, p) => s + Math.pow(((p.x / 600) < tA ? b : c) - (250 - p.y) / 250, 2), 0);
            if (err < best.err) best = { a: tA, b, c, err };
        }
        this.params = best;
    }
}

class NaiveBayesModel extends BaseModel {
    constructor() { super(); this.isClassifier = true; this.stats = {}; }
    fit(points) {
        [0, 1].forEach(c => {
            const vals = points.filter(p => p.label === c).map(p => p.x / 600);
            if (vals.length === 0) { this.stats[c] = { m: 0.5, s: 0.5, p: 0 }; return; }
            const m = vals.reduce((a, b) => a + b, 0) / vals.length;
            const v = vals.reduce((a, b) => a + Math.pow(b - m, 2), 0) / vals.length;
            this.stats[c] = { m, s: Math.sqrt(v || 0.01), p: vals.length / points.length };
        });
    }
    predict(x) {
        const probs = [0, 1].map(c => {
            const { m, s, p } = this.stats[c];
            if (p === 0) return 0;
            return p * (1 / (Math.sqrt(2 * Math.PI) * s)) * Math.exp(-Math.pow(x - m, 2) / (2 * s * s));
        });
        return probs[1] > probs[0] ? 1 : 0;
    }
}

class KNNModel extends BaseModel {
    constructor(k = 5) {
        super();
        this.isClassifier = true;
        this.k = k;
        this.data = [];
    }
    fit(points) { this.data = points; }
    predict(x) {
        if (!this.data.length) return 0;
        const dists = this.data.map(p => ({
            d: Math.abs(x - (p.x / 600)),
            label: p.label
        })).sort((a, b) => a.d - b.d);
        const neighbors = dists.slice(0, this.k);
        const votes = neighbors.reduce((acc, n) => { acc[n.label]++; return acc; }, { 0: 0, 1: 0 });
        return votes[1] > votes[0] ? 1 : 0;
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
        this.inputs = { spread: this.el.querySelector('.spread-input'), outlier: this.el.querySelector('.outlier-input') };
        this.el.querySelector('.model-title').textContent = type;
        this.el.querySelector('.model-formula').textContent = formula;
        container.appendChild(this.el);
        this.model = modelClass;
        this.points = [];
        this.el.querySelector('.reroll-btn').onclick = () => this.generate();
        Object.values(this.inputs).forEach(i => i.oninput = () => this.generate());
        this.generate();
    }
    generate() {
        const spr = this.inputs.spread.value / 100, out = this.inputs.outlier.value / 100;
        this.points = [];
        const mode = ['clusters', 'gap', 'hetero', 'normal'][Math.floor(Math.random() * 4)];
        for (let i = 0; i < 70; i++) {
            let xN = Math.random(), yN, label = null;
            if (this.isClassifier) {
                label = Math.random() > 0.5 ? 1 : 0;
                xN = Math.max(0, Math.min(1, (label === 0 ? 0.3 : 0.7) + (Math.random() - 0.5) * spr * 1.5));
                if (Math.random() < out) label = 1 - label;
                yN = (label === 0 ? 0.25 : 0.75) + (Math.random() - 0.5) * 0.2;
            } else {
                if (mode === 'gap' && xN > 0.4 && xN < 0.6) xN += 0.25;
                if (mode === 'clusters') xN = Math.floor(xN * 5) / 5 + (Math.random() * 0.05);
                switch (this.type) {
                    case 'Linear': yN = 0.5 * xN + 0.2; break;
                    case 'Polynomial': yN = 2.5 * Math.pow(xN - 0.5, 2) + 0.2; break;
                    case 'Exponential': yN = 0.15 * Math.exp(1.8 * xN) + 0.1; break;
                    case 'Logarithmic': yN = 0.5 + 0.2 * Math.log(xN + 0.01); break;
                    case 'Periodic': yN = 0.2 * Math.sin(10 * xN) + 0.5; break;
                    case 'Step': yN = xN < 0.5 ? 0.2 : 0.8; break;
                    case 'Logistic': yN = 1 / (1 + Math.exp(-(xN - 0.5) * 10)); break;
                }
                if (Math.random() < out) yN = Math.random();
                else yN += (Math.random() - 0.5) * (mode === 'hetero' ? xN * spr : spr);
            }
            this.points.push({ x: xN * 600, y: 250 - (Math.max(0, Math.min(1, yN)) * 250), label });
        }
        this.model.fit(this.points);
        const m = this.model.getMetrics(this.points);
        [m.m1, m.m2, m.m3].forEach((v, i) => this.labels[i].textContent = v);
        this.draw();
    }
    draw() {
        this.ctx.clearRect(0, 0, 600, 250);
        if (this.isClassifier) {
            for (let x = 0; x < 600; x += 4) {
                this.ctx.fillStyle = this.model.predict(x / 600) === 1 ? 'rgba(88, 166, 255, 0.15)' : 'rgba(255, 123, 114, 0.15)';
                this.ctx.fillRect(x, 0, 4, 250);
            }
        }
        this.points.forEach(p => {
            this.ctx.fillStyle = this.isClassifier ? (p.label === 1 ? '#58a6ff' : '#ff7b72') : '#58a6ff';
            this.ctx.beginPath(); this.ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2); this.ctx.fill();
        });
        if (!this.isClassifier) {
            this.ctx.strokeStyle = '#ff7b72'; this.ctx.lineWidth = 3; this.ctx.beginPath();
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
new ModelViz('Naive Bayes', 'P(C|x) == P(x|C)P(C)', container, new NaiveBayesModel(), true);
new ModelViz('KNN', 'k = 5', container, new KNNModel(5), true);