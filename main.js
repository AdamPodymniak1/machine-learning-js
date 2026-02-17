class BaseRegressor {
    constructor(type) {
        this.type = type;
        this.params = { a: 0.5, b: 0.5, c: 0.5, d: 0.5 };
        this.lr = 0.15;
        this.iters = 3500;
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    predict(x) {
        const { a, b, c, d } = this.params;
        switch (this.type) {
            case 'Linear': return a * x + b;
            case 'Polynomial': return a * Math.pow(x - 0.5, 2) + b * x + c;
            case 'Exponential': return a * Math.exp(b * x) + c;
            case 'Logarithmic': return a + b * Math.log(x + 0.05);
            case 'Periodic': return a * Math.sin(b * x + c) + d;
            case 'Step': return x < a ? b : c;
            case 'Logistic': return this.sigmoid((x - a) * b * 10);
        }
    }

    fit(points) {
        if (this.type === 'Step') {
            this.fitStep(points);
            return;
        }

        this.params = { a: 0.2, b: 1, c: 0.5, d: 0.5 };
        if (this.type === 'Periodic') this.params.b = 15;
        if (this.type === 'Logistic') { this.params.a = 0.5; this.params.b = 5; }

        for (let i = 0; i < this.iters; i++) {
            let da = 0, db = 0, dc = 0, dd = 0;
            for (let p of points) {
                const x = p.x / 600;
                const y = (250 - p.y) / 250;
                const pred = this.predict(x);
                const err = pred - y;

                if (this.type === 'Linear') { da += err * x; db += err; }
                else if (this.type === 'Polynomial') { da += err * Math.pow(x - 0.5, 2); db += err * x; dc += err; }
                else if (this.type === 'Exponential') {
                    da += err * Math.exp(this.params.b * x);
                    db += err * this.params.a * x * Math.exp(this.params.b * x);
                    dc += err;
                } else if (this.type === 'Logarithmic') { da += err; db += err * Math.log(x + 0.05); }
                else if (this.type === 'Periodic') {
                    da += err * Math.sin(this.params.b * x + this.params.c);
                    db += err * this.params.a * x * Math.cos(this.params.b * x + this.params.c);
                    dc += err * this.params.a * Math.cos(this.params.b * x + this.params.c);
                    dd += err;
                } else if (this.type === 'Logistic') {
                    const s = pred * (1 - pred);
                    da += err * s * (-this.params.b * 10);
                    db += err * s * (x - this.params.a) * 10;
                }
            }
            const n = points.length;
            this.params.a -= (da / n) * this.lr;
            this.params.b -= (db / n) * this.lr;
            this.params.c -= (dc / n) * this.lr;
            this.params.d -= (dd / n) * this.lr;
        }
    }

    fitStep(points) {
        let bestA = 0.5, bestB = 0, bestC = 0, minErr = Infinity;
        for (let tA = 0.1; tA < 0.9; tA += 0.05) {
            const left = points.filter(p => (p.x / 600) < tA);
            const right = points.filter(p => (p.x / 600) >= tA);
            if (!left.length || !right.length) continue;
            const avgB = left.reduce((s, p) => s + (250 - p.y) / 250, 0) / left.length;
            const avgC = right.reduce((s, p) => s + (250 - p.y) / 250, 0) / right.length;
            let err = 0;
            points.forEach(p => {
                const pred = (p.x / 600) < tA ? avgB : avgC;
                err += Math.pow(pred - (250 - p.y) / 250, 2);
            });
            if (err < minErr) { minErr = err; bestA = tA; bestB = avgB; bestC = avgC; }
        }
        this.params = { a: bestA, b: bestB, c: bestC };
    }

    getMetrics(points) {
        let sse = 0, sae = 0, sst = 0;
        const n = points.length;
        const yVals = points.map(p => (250 - p.y) / 250);
        const yMean = yVals.reduce((a, b) => a + b, 0) / n;

        for (let p of points) {
            const x = p.x / 600;
            const y = (250 - p.y) / 250;
            const pred = this.predict(x);
            const err = pred - y;
            sse += err * err;
            sae += Math.abs(err);
            sst += Math.pow(y - yMean, 2);
        }
        return {
            mse: (sse / n).toFixed(5),
            mae: (sae / n).toFixed(5),
            r2: (sst === 0 ? 0 : 1 - (sse / sst)).toFixed(3)
        };
    }
}

class ModelViz {
    constructor(type, formula, container) {
        this.type = type;
        const temp = document.getElementById('model-template');
        const clone = temp.content.cloneNode(true);
        this.el = clone.querySelector('.model-card');
        this.canvas = clone.querySelector('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.mseLabel = this.el.querySelector('.mse-value');
        this.maeLabel = this.el.querySelector('.mae-value');
        this.r2Label = this.el.querySelector('.r2-value');
        this.spreadInput = this.el.querySelector('.spread-input');
        this.outlierInput = this.el.querySelector('.outlier-input');
        this.rerollBtn = this.el.querySelector('.reroll-btn');

        this.el.querySelector('.model-title').textContent = type;
        this.el.querySelector('.model-formula').textContent = formula;
        container.appendChild(this.el);

        this.model = new BaseRegressor(type);
        this.points = [];
        this.initEvents();
        this.generate();
    }

    initEvents() {
        this.rerollBtn.onclick = () => this.generate();
        this.spreadInput.oninput = () => this.generate();
        this.outlierInput.oninput = () => this.generate();
    }

    generate() {
        const spread = this.spreadInput.value / 100;
        const outlierChance = this.outlierInput.value / 100;
        this.points = [];
        const mode = ['clusters', 'gap', 'hetero', 'normal'][Math.floor(Math.random() * 4)];

        for (let i = 0; i < 70; i++) {
            let xN = Math.random();
            if (mode === 'gap' && xN > 0.4 && xN < 0.6) xN += 0.25;
            if (mode === 'clusters') xN = Math.floor(xN * 5) / 5 + (Math.random() * 0.05);

            let yN;
            switch (this.type) {
                case 'Linear': yN = 0.5 * xN + 0.2; break;
                case 'Polynomial': yN = 2.5 * Math.pow(xN - 0.5, 2) + 0.2; break;
                case 'Exponential': yN = 0.1 * Math.exp(2 * xN) + 0.1; break;
                case 'Logarithmic': yN = 0.5 + 0.2 * Math.log(xN + 0.01); break;
                case 'Periodic': yN = 0.2 * Math.sin(10 * xN) + 0.5; break;
                case 'Step': yN = xN < 0.5 ? 0.2 : 0.8; break;
                case 'Logistic': yN = xN < 0.5 ? 0.1 : 0.9; break;
            }

            if (Math.random() < outlierChance) yN = Math.random();
            else yN += (Math.random() - 0.5) * (mode === 'hetero' ? xN * spread : spread);

            yN = Math.max(0, Math.min(1, yN));
            this.points.push({ x: xN * 600, y: 250 - (yN * 250) });
        }
        this.model.fit(this.points);
        const m = this.model.getMetrics(this.points);
        this.mseLabel.textContent = m.mse;
        this.maeLabel.textContent = m.mae;
        this.r2Label.textContent = m.r2;
        this.draw();
    }

    draw() {
        this.ctx.clearRect(0, 0, 600, 250);
        this.points.forEach(p => {
            this.ctx.fillStyle = '#58a6ff';
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.strokeStyle = '#ff7b72';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        for (let x = 0; x <= 600; x += 2) {
            const yN = this.model.predict(x / 600);
            this.ctx.lineTo(x, 250 - (yN * 250));
        }
        this.ctx.stroke();
    }
}

const container = document.getElementById('models-container');
const types = [
    ['Linear', 'y = ax + b'],
    ['Polynomial', 'y = a(x-0.5)^2 + bx + c'],
    ['Exponential', 'y = ae^(bx) + c'],
    ['Logarithmic', 'y = a + b * ln(x)'],
    ['Periodic', 'y = a * sin(bx + c) + d'],
    ['Step', 'y = (x < a) ? b : c'],
    ['Logistic', 'y = 1 / (1 + e^-z)']
];
types.forEach(t => new ModelViz(t[0], t[1], container));