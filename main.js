class BaseRegressor {
    constructor(type) {
        this.type = type;
        this.params = { a: 0.5, b: 0.5, c: 0.5 };
        this.lr = 0.15;
        this.iters = 3000;
    }

    predict(x) {
        const { a, b, c } = this.params;
        switch (this.type) {
            case 'Linear': return a * x + b;
            case 'Polynomial': return a * Math.pow(x, 2) + b * x + c;
            case 'Exponential': return a * Math.exp(b * x);
            case 'Logarithmic': return a + b * Math.log(x + 0.05);
        }
    }

    fit(points) {
        this.params = { a: 0.1, b: 0.1, c: 0.1 };
        for (let i = 0; i < this.iters; i++) {
            let da = 0, db = 0, dc = 0;
            for (let p of points) {
                const x = p.x / 600;
                const y = (250 - p.y) / 250;
                const err = this.predict(x) - y;
                if (this.type === 'Linear') { da += err * x; db += err; }
                else if (this.type === 'Polynomial') { da += err * x * x; db += err * x; dc += err; }
                else if (this.type === 'Exponential') {
                    da += err * Math.exp(this.params.b * x);
                    db += err * this.params.a * x * Math.exp(this.params.b * x);
                } else if (this.type === 'Logarithmic') {
                    da += err; db += err * Math.log(x + 0.05);
                }
            }
            this.params.a -= (da / points.length) * this.lr;
            this.params.b -= (db / points.length) * this.lr;
            this.params.c -= (dc / points.length) * this.lr;
        }
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
        this.el.querySelector('.model-title').textContent = type;
        this.el.querySelector('.model-formula').textContent = formula;
        container.appendChild(this.el);
        this.model = new BaseRegressor(type);
        this.points = [];
    }

    generate(spread, outlierFactor) {
        this.points = [];
        const count = 50;
        const randA = (Math.random() - 0.5) * 1.5;
        const randB = (Math.random() - 0.5) * 1.2;
        const randC = Math.random() * 0.5;

        for (let i = 0; i < count; i++) {
            const xNorm = Math.random();
            let yNorm;
            if (this.type === 'Linear') yNorm = randA * xNorm + randC + 0.2;
            else if (this.type === 'Polynomial') yNorm = randA * Math.pow(xNorm, 2) + randB * xNorm + 0.5;
            else if (this.type === 'Exponential') yNorm = 0.2 * Math.exp(randB * xNorm) + randC;
            else if (this.type === 'Logarithmic') yNorm = 0.4 + (randB * 0.3) * Math.log(xNorm + 0.1);

            if (Math.random() < outlierFactor / 100) yNorm = Math.random();
            else yNorm += (Math.random() - 0.5) * (spread / 100);

            yNorm = Math.max(0, Math.min(1, yNorm));
            this.points.push({ x: xNorm * 600, y: 250 - (yNorm * 250) });
        }
        this.model.fit(this.points);
        this.draw();
    }

    draw() {
        this.ctx.clearRect(0, 0, 600, 250);
        this.ctx.fillStyle = '#4a90e2';
        this.points.forEach(p => {
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.strokeStyle = '#ff4757';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        for (let x = 0; x <= 600; x += 4) {
            const yNorm = this.model.predict(x / 600);
            this.ctx.lineTo(x, 250 - (yNorm * 250));
        }
        this.ctx.stroke();
    }
}

const container = document.getElementById('models-container');
const models = [
    new ModelViz('Linear', 'y = ax + b', container),
    new ModelViz('Exponential', 'y = a * e^(bx)', container),
    new ModelViz('Logarithmic', 'y = a + b * ln(x)', container),
    new ModelViz('Polynomial', 'y = ax² + bx + c', container),
];

const updateAll = () => {
    const s = document.getElementById('globalSpread').value;
    const o = document.getElementById('globalOutliers').value;
    models.forEach(m => m.generate(s, o));
};

document.getElementById('rerollAll').onclick = updateAll;
document.querySelectorAll('input').forEach(i => i.oninput = updateAll);
updateAll();