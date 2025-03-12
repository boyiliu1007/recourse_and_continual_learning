def animate_all(self, frames: int = 120, fps: int = 10, *, inplace: bool = False):
        fig, (ax0, ax1, ax2) = self.draw_all()  # Initial figure setup
        self._ct_test = None  # Ensure contour plot is initialized

        def init():
            return *ax0.patches, *ax1.collections, *ax2.collections

        def func(frame):
            nonlocal fig, ax0, ax1, ax2  # Ensure variables persist across frames

            fig.suptitle(f'No. {frame}', ha='left', x=0.01, size='small')

            if frame == 0:
                return ()

            self.update(self.model, self.train, self.sample)

            # 🔹 Save a snapshot every 10 rounds with `ax0` (User Response) on top
            if frame % 10 == 0 or frame == 1:
                fig_snapshot, (snap_ax0, snap_ax1) = plt.subplots(
                    2, 1,  # 2 rows, 1 column
                    sharex=True,
                    sharey=True,
                    figsize=(6, 8),
                    layout='compressed'
                )

                # 🎨 Re-draw the scatter plots with the correct order
                prop = dict(cmap=self.cmap, s=40, vmin=0., vmax=1., lw=0.8, edgecolor='w')

                self._sc_train = snap_ax0.scatter(
                    *pca.transform(self.train.x).T,
                    c=self.train.y,
                    **prop
                )
                snap_ax0.legend(*self._sc_train.legend_elements(), loc='upper right', title='True label')
                snap_ax0.set_xlabel('pca0')
                snap_ax0.set_ylabel('pca1')
                snap_ax0.set_title('User Responded Dataset')

                y_pred = self.model(self.test.x).flatten().greater(0.5)

                self._sc_test = snap_ax1.scatter(
                    *pca.transform(self.test.x).T,
                    c=y_pred,
                    **prop
                )
                snap_ax1.legend(*self._sc_test.legend_elements(), loc='lower right', title='Predicted')
                snap_ax1.set_xlabel('pca0')
                snap_ax1.set_ylabel('pca1')
                snap_ax1.set_title('Test Dataset')

                # 🔹 Add model decision boundary to test dataset (snap_ax1)
                x0, x1 = snap_ax1.get_xlim()
                y0, y1 = snap_ax1.get_ylim()
                n = 100
                xy = np.mgrid[x0:x1:n * 1j, y0:y1:n * 1j]
                z = pca.inverse_transform(xy.reshape(2, n * n).T)
                z = pt.tensor(z, dtype=pt.float)

                with pt.no_grad():
                    z: pt.Tensor = self.model(z)
                z = z.view(n, n)

                snap_ax1.contourf(
                    *xy, z, 10,
                    cmap='RdYlBu_r',
                    vmin=0,
                    vmax=1,
                    alpha=0.9,
                    zorder=0,
                )

                # 📸 Save the snapshot
                snapshot_path = os.path.join(
                    self.save_directory, f"scatter_round_{frame}.png"
                )
                fig_snapshot.savefig(snapshot_path, dpi=300)
                plt.close(fig_snapshot)  # Close to free memory
                print(f"✅ Snapshot saved: {snapshot_path}")

            y = self.test.y.flatten()
            n = self.test.x.shape[0]
            m = n - pt.count_nonzero(y)

            with pt.no_grad():
                y_prob: pt.Tensor = self.model(self.test.x)

            y_prob = y_prob.flatten()
            y_pred = y_prob.greater(0.5)

            for b, r in zip(self._hist, (y_prob[y.argsort()][:m], y_prob[y.argsort()][m:])):
                height, _ = np.histogram(r, self._bins)
                for r, h in zip(b.patches, height * (100 / n)):
                    r.set_height(h)

            self._sc_train.set_offsets(pca.transform(self.train.x))
            self._sc_train.set_array(self.train.y.flatten())

            self.js_divergence(self._sc_train.get_offsets(), self._sc_train.get_array())
            self._sc_test.set_array(y_pred)

            ax1.relim()
            ax1.autoscale_view()

            # 🔹 Update model decision boundary for animation
            if self._ct_test:
                for c in self._ct_test.collections:
                    c.remove()

            x0, x1 = ax1.get_xlim()
            y0, y1 = ax1.get_ylim()
            n = 32
            xy = np.mgrid[x0: x1: n * 1j, y0: y1: n * 1j]
            z = pca.inverse_transform(xy.reshape(2, n * n).T)
            z = pt.tensor(z, dtype=pt.float)

            with pt.no_grad():
                z: pt.Tensor = self.model(z)

            z = z.view(n, n)

            self._ct_test = ax2.contourf(
                *xy, z, 10,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                alpha=0.9,
                zorder=0,
            )

            return *ax0.patches, *ax1.collections, *ax2.collections

        ani = FuncAnimation(
            fig, func, frames, init,
            interval=1000 // fps,
            repeat=False,
            blit=True,
            cache_frame_data=False
        )

        return ani